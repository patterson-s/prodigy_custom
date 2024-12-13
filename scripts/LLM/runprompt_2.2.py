import os
import json
import pandas as pd
import cohere
from pathlib import Path
from typing import Dict, Any, List

def list_prompt_files(directory: str, extension: str) -> List[str]:
    """List all prompt files in the prompts directory."""
    return [f for f in os.listdir(directory) if f.endswith(extension)]

def choose_prompt(files: List[str]) -> str:
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

def load_prompt_and_metadata(prompts_dir: str, prompt_file: str) -> tuple:
    """Load prompt content and its associated metadata."""
    # Load prompt
    prompt_path = os.path.join(prompts_dir, prompt_file)
    with open(prompt_path, 'r', encoding='utf-8') as file:
        prompt = file.read()
    
    # Load metadata (assuming same name with .json extension)
    metadata_file = prompt_file.replace('.txt', '_meta.json')
    metadata_path = os.path.join(prompts_dir, metadata_file)
    with open(metadata_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)
    
    return prompt, metadata

def process_dataset(spec: Dict[str, Any]) -> pd.DataFrame:
    """Load and process a single dataset according to specifications."""
    # Load dataset based on format
    df = pd.read_json(spec['path'], lines=True) if spec['format'] == 'jsonl' else pd.read_csv(spec['path'])
    
    # Apply subset if specified
    if 'subset' in spec:
        query = ' & '.join([f"{k} == {repr(v)}" for k, v in spec['subset'].items()])
        df = df.query(query)
    
    # Select specified variables if they exist
    if 'variables' in spec:
        available_cols = set(df.columns)
        requested_cols = set(spec['variables'])
        valid_cols = list(available_cols.intersection(requested_cols))
        
        if valid_cols:
            df = df[valid_cols]
        else:
            print(f"Warning: No specified variables found in dataset. Using all columns.")
    
    return df

def run_prompt(co_client: cohere.Client, prompt: str, metadata: Dict[str, Any]) -> str:
    """Run the prompt with Cohere API using parameters from metadata."""
    try:
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
    except Exception as e:
        print(f"Error running prompt: {str(e)}")
        return f"ERROR: {str(e)}"

def format_sample_result(result_row: Dict[str, Any], output: str) -> str:
    """Format sample result for display."""
    sample = "\n=== SAMPLE RESULT ===\n"
    
    # Show core fields if they exist
    core_fields = ['doc_id', 'chunk_id', 'source', 'target_iso', 'year']
    for field in core_fields:
        if field in result_row:
            sample += f"{field}: {result_row[field]}\n"
    
    # Add the output
    sample += f"Output:\n{output}\n"
    sample += "===================\n"
    return sample

def preview_prompt(prompt_template: str, combined_data: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    """Show a complete example of how the prompt will look when formatted."""
    print("\n=== PROMPT PREVIEW ===")
    print("Using first row of data to demonstrate prompt formatting:\n")
    
    # Get first row
    first_row = combined_data.iloc[0]
    
    # Format prompt with first row data
    prompt_vars = {var: first_row[var] for var in metadata['join']['variables_in_prompt']}
    prompt_instance = prompt_template.format(**prompt_vars)
    
    print("Raw data used:")
    for var, value in prompt_vars.items():
        print(f"{var}: {value}")
    
    print("\nFormatted prompt:")
    print("-" * 80)
    print(prompt_instance)
    print("-" * 80)
    
    proceed = input("\nWould you like to proceed with processing? (y/n): ")
    if proceed.lower() != 'y':
        print("Operation cancelled")
        exit()

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
        
    # Preview prompt before processing
    preview_prompt(prompt_template, combined_data, metadata)

    # Run prompts
    results = []
    total_rows = len(combined_data)
    print(f"\nRunning prompts on {total_rows} rows...")
    
    for idx, row in combined_data.iterrows():
        print(f"\nProcessing row {idx + 1} of {total_rows}...")
        
        # Format prompt with row data
        try:
            prompt_vars = {var: row[var] for var in metadata['join']['variables_in_prompt']}
            prompt_instance = prompt_template.format(**prompt_vars)
        except KeyError as e:
            print(f"Error formatting prompt: Missing variable {e}")
            continue
        
        # Run prompt
        result = run_prompt(co_client, prompt_instance, metadata)
        
        # Prepare result row - keep all original data
        result_row = row.to_dict()
        result_row[f"{selected_prompt.replace('.txt', '')}_output"] = result
        results.append(result_row)

        # Print sample every 10 entries
        if (idx + 1) % 10 == 0:
            print(format_sample_result(result_row, result))

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