import os
import json
import pandas as pd
import cohere
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def load_prompt_and_metadata(prompt_path: str) -> str:
    """Load prompt template from file."""
    with open(prompt_path, 'r') as file:
        return file.read()

def group_by_document_target(data: List[dict]) -> Dict[Tuple[str, str, str, int], List[str]]:
    """Group inductive outputs by document ID, target ISO, source country, and year."""
    grouped_data = defaultdict(list)
    for item in data:
        key = (item['doc_id'], item['target_iso'], item['source_country'], item['year'])
        grouped_data[key].append(item['inductive_01_output'])
    return grouped_data

def create_analysis_prompt(template: str, source: str, target: str, year: int, 
                         summaries: List[str]) -> str:
    """Create a prompt for analyzing a specific document-target combination."""
    # Format the summaries first
    formatted_summaries = "\n".join(f"- {summary}" for summary in summaries)
    
    # Replace placeholders in template
    return template.replace("{source}", source)\
                  .replace("{target}", target)\
                  .replace("{year}", str(year))\
                  .replace("{summaries}", formatted_summaries)

def run_prompt(co_client, prompt: str) -> str:
    """Run the prompt with Cohere API."""
    response = co_client.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        k=0,
        p=0.75,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop_sequences=[],
        return_likelihoods="NONE"
    )
    return response.generations[0].text.strip()

def process_documents(input_file: str, prompt_path: str, output_dir: str):
    """Process all documents and generate relationship analyses."""
    # Initialize Cohere client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    co_client = cohere.Client(api_key)

    # Load prompt template
    prompt_template = load_prompt_and_metadata(prompt_path)

    # Read input data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    # Group by document and target
    grouped_data = group_by_document_target(data)

    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each group
    results = []
    total_groups = len(grouped_data)
    
    print(f"Processing {total_groups} document-target combinations...")
    
    for i, ((doc_id, target_iso, source_country, year), summaries) in enumerate(grouped_data.items(), 1):
        print(f"Processing {i}/{total_groups}: {doc_id} - {target_iso}")
        
        # Create prompt
        prompt = create_analysis_prompt(
            prompt_template, 
            source_country, 
            target_iso, 
            year,
            summaries
        )

        # Run prompt
        try:
            result = run_prompt(co_client, prompt)
            
            # Store result
            results.append({
                'doc_id': doc_id,
                'target_iso': target_iso,
                'source_country': source_country,
                'year': year,
                'analysis': result
            })
            
        except Exception as e:
            print(f"Error processing {doc_id} - {target_iso}: {str(e)}")
            continue

        # Save intermediate results every 10 documents
        if i % 10 == 0:
            save_results(results, output_dir / f'intermediate_results_{i}.jsonl')

    # Save final results
    save_results(results, output_dir / 'final_results.jsonl')
    print(f"\nProcessing complete! Results saved to {output_dir}")

def save_results(results: List[dict], output_file: Path):
    """Save results to a JSONL file."""
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    # File paths
    input_file = r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/prompt_output/inductive_01/inductive_01_8dec24.jsonl"
    prompt_path = r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/prompts/inductive_relationship_01.txt"
    output_dir = r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/prompt_output/inductive_relationship_01"

    # Run processing
    process_documents(input_file, prompt_path, output_dir)