import os
import json
import pandas as pd
import cohere
from pathlib import Path

def list_prompt_files(directory, extension):
    """List all prompt files in the prompts directory."""
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def choose_prompt(files):
    """Display available prompt files and allow user to select one."""
    print("Available Prompts:")
    for idx, file in enumerate(files, start=1):
        print(f"{idx}: {file}")

    while True:
        try:
            choice = int(input("Enter the number of the prompt you want to use: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def load_prompt_and_metadata(prompts_dir, prompt_file):
    """Load prompt content and its associated metadata."""
    # Load prompt
    prompt_path = os.path.join(prompts_dir, prompt_file)
    with open(prompt_path, 'r') as file:
        prompt = file.read()
    
    # Load metadata (assuming same name with .json extension)
    metadata_file = prompt_file.replace('.txt', '_meta.json')
    metadata_path = os.path.join(prompts_dir, metadata_file)
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)
    
    return prompt, metadata

def process_dataset(spec):
    """Load and process a single dataset according to specifications."""
    # Load dataset based on format
    df = pd.read_json(spec['path'], lines=True) if spec['format'] == 'jsonl' else pd.read_csv(spec['path'])
    
    # Apply subset if specified
    if 'subset' in spec:
        query = ' & '.join([f"{k} == {repr(v)}" for k, v in spec['subset'].items()])
        df = df.query(query)
    
    # Select specified variables
    return df[spec['variables']]

def run_prompt(co_client, prompt, metadata):
    """Run the prompt with Cohere API using parameters from metadata."""
    response = co_client.generate(
        model=metadata["model"],
        prompt=prompt,
        max_tokens=metadata["max_tokens"],
        temperature=metadata["temperature"],
        k=metadata["k"],
        p=metadata["p"],
        frequency_penalty=metadata["frequency_penalty"],
        presence_penalty=metadata["presence_penalty"],
        stop_sequences=metadata["stop_sequences"],
        return_likelihoods=metadata["return_likelihoods"]
    )
    return response.generations[0].text.strip()

def main():
    # Initialize the Cohere client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("Cohere API key not found. Please ensure it's set in the environment.")
    co_client = cohere.Client(api_key)

    # Set up prompts directory
    prompts_dir = r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/prompts"

    # List and choose prompt file
    prompt_files = list_prompt_files(prompts_dir, ".txt")
    if not prompt_files:
        print("No prompt files found.")
        return
    
    selected_prompt = choose_prompt(prompt_files)
    prompt_template, metadata = load_prompt_and_metadata(prompts_dir, selected_prompt)

    # Process primary dataset
    print("\nProcessing primary dataset...")
    primary_df = process_dataset(metadata['datasets']['primary'])

    # Process and join secondary dataset if it exists
    if 'secondary' in metadata['datasets']:
        print("Processing secondary dataset...")
        secondary_df = process_dataset(metadata['datasets']['secondary'])
        join_key = metadata['join']['key']
        combined_data = pd.merge(primary_df, secondary_df, on=join_key)
    else:
        combined_data = primary_df

    # Run prompts
    results = []
    total_rows = len(combined_data)
    print(f"\nRunning prompts on {total_rows} rows...")
    
    for idx, row in combined_data.iterrows():
        print(f"\nProcessing row {idx + 1} of {total_rows}...")
        
        # Format prompt with row data
        prompt_vars = {var: row[var] for var in metadata['join']['variables_in_prompt']}
        prompt_instance = prompt_template.format(**prompt_vars)
        
        # Run prompt
        result = run_prompt(co_client, prompt_instance, metadata)
        
        # Prepare result row
        result_row = row.to_dict()
        result_row[f"{selected_prompt.replace('.txt', '')}_output"] = result
        results.append(result_row)

    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    
    # Create output directory if it doesn't exist
    os.makedirs(metadata['output']['directory'], exist_ok=True)
    
    # Save results
    output_path = os.path.join(metadata['output']['directory'], 'results.jsonl')
    results_df.to_json(output_path, orient='records', lines=True)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()