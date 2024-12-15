import os
import cohere
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import shutil

def create_prompt(source, target, year, text):
    return (
        "You are an expert in analyzing diplomatic speech and international relations. "
        "Your task is to identify explicit references to international institutions in diplomatic speech. "
        "You will receive an excerpt from a UN General Assembly speech.\n\n"
        f"Analyze this excerpt from a {year} UN General Assembly speech:\n"
        f"{text}\n\n"
        "If you find explicit mentions of international institutions, provide:\n"
        "1. The tag [institution]\n"
        "2. The exact institutional reference as quoted in the text, introduced by \"descriptive: \"\n\n"
        "Example formats:\n"
        "For \"The Security Council must act...\":\n"
        "tag: [institution]; descriptive: \"Security Council\"\n\n"
        "For multiple institutions:\n"
        "tag: [institution]; descriptive: \"Security Council\"\n"
        "tag: [institution]; descriptive: \"International Court of Justice\"\n\n"
        "Only include institutions that are explicitly mentioned in the text. "
        "If no international institutions are explicitly referenced, respond with NA.\n\n"
        "Notes:\n"
        "- Use exact quotes for the descriptive text\n"
        "- Do not infer or expand institutional references beyond what is directly stated\n"
        "- Each distinct institution should be listed separately\n"
        "- 'the Council' refers to the 'Security Council'; 'the Assembly' refers to the 'General Assembly'\n"
        "- Abbreviations (like UN, ICJ) should be recorded as they appear in the text"
    )

def analyze_institutions(co_client, source, target, year, summaries):
    try:
        prompt = create_prompt(source, target, year, summaries)
        
        # Make the chat API call
        response = co_client.chat(
            model="command-r7b-12-2024",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract the text content from the response
        # The response contains a list of content items, we want the text from the first one
        analysis = response.message.content[0].text
        return analysis, prompt
        
    except Exception as e:
        print(f"Debug - Error type: {type(e)}")
        print(f"Debug - Error message: {str(e)}")
        return f"Error: {str(e)}", prompt

def process_batch(batch_df, co_client, source_col, target_col, year_col, text_col):
    results = []
    for _, row in batch_df.iterrows():
        analysis, _ = analyze_institutions(
            co_client,
            row[source_col],
            row[target_col],
            row[year_col],
            row[text_col]
        )
        result = {
            'source': row[source_col],
            'target': row[target_col],
            'year': row[year_col],
            'text': row[text_col],
            'analysis': analysis
        }
        results.append(result)
    return results

def main():
    # Initialize Cohere client with new V2 client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    co_client = cohere.ClientV2(api_key)

    # Get input file path
    while True:
        input_path = input("\nEnter the path to your input CSV file: ").strip()
        if os.path.exists(input_path):
            break
        print("File not found. Please try again.")

    # Load dataset and show columns
    df = pd.read_csv(input_path)
    print("\nAvailable columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")

    # Get column selections
    def get_column_selection(prompt):
        while True:
            try:
                selection = int(input(prompt)) - 1
                if 0 <= selection < len(df.columns):
                    return df.columns[selection]
                print("Invalid selection. Please try again.")
            except ValueError:
                print("Please enter a number.")

    source_col = get_column_selection("\nSelect the source country column number: ")
    target_col = get_column_selection("Select the target country column number: ")
    year_col = get_column_selection("Select the year column number: ")
    text_col = get_column_selection("Select the text column number: ")

    # Handle self-mentions
    total_rows = len(df)
    self_mentions = len(df[df['iso'] == df['ISO_Code']])
    print(f"\nFound {self_mentions} self-mentions out of {total_rows} total rows.")
    include_self = input("Include self-mentions? (y/n): ").lower() == 'y'
    
    if not include_self:
        df = df[df['iso'] != df['ISO_Code']]
        print(f"Filtered dataset now contains {len(df)} rows")

    # Set up output path
    input_path_obj = Path(input_path)
    default_output = input_path_obj.parent / f"{input_path_obj.stem}_institutions.jsonl"
    print(f"\nDefault output path: {default_output}")
    if input("Use this path? (y/n): ").lower() != 'y':
        output_path = input("Enter custom output path: ")
    else:
        output_path = default_output

    # Create temp directory
    temp_dir = Path(output_path).parent / "temp"
    temp_dir.mkdir(exist_ok=True)

    # Get batch size
    while True:
        try:
            batch_size = int(input("\nEnter batch size (recommended: 500): "))
            if batch_size > 0:
                break
            print("Batch size must be positive.")
        except ValueError:
            print("Please enter a valid number.")

    # Show example prompt and its output
    first_row = df.iloc[0]
    analysis, prompt = analyze_institutions(
        co_client,
        first_row[source_col],
        first_row[target_col],
        first_row[year_col],
        first_row[text_col]
    )
    
    print("\nExample prompt for first row:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    print("\nCohere response:")
    print("-" * 80)
    print(analysis)
    print("-" * 80)
    
    if input("\nContinue with this output format? (y/n): ").lower() != 'y':
        print("Exiting...")
        return

    # Process in batches
    total_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
    print(f"\nProcessing {len(df)} rows in {total_batches} batches...")

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(df))
        
        print(f"\nProcessing batch {batch_num + 1}/{total_batches}")
        batch_df = df.iloc[start_idx:end_idx]
        
        results = process_batch(batch_df, co_client, source_col, target_col, year_col, text_col)
        
        # Save batch results
        temp_file = temp_dir / f"batch_{batch_num}.jsonl"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"Saved batch {batch_num + 1} results to {temp_file}")

    # Combine all temp files
    print("\nCombining temporary files...")
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for temp_file in sorted(temp_dir.glob('batch_*.jsonl')):
            with open(temp_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())

    # Cleanup
    print("Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    print(f"\nProcessing complete! Results saved to: {output_path}")

if __name__ == "__main__":
    main()