import cohere
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict

def get_cohere_client():
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    return cohere.ClientV2(api_key)

def save_to_jsonl(data, output_file):
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def subset_data(data):
    """Subset the dataset based on user input."""
    print("\nWould you like to subset the data? Options: 'year', 'random x', 'no subset'")
    subset_option = input("Enter your choice: ").strip().lower()

    if subset_option == "year":
        if 'year' not in data.columns:
            print("Column 'year' not found in dataset.")
            year_column = input("Please enter the correct column name for 'year': ").strip()
        else:
            year_column = 'year'

        year = input("Enter the year to filter by: ").strip()
        if year_column in data.columns:
            return data[data[year_column] == int(year)]
        else:
            print(f"Error: Column '{year_column}' not found. Proceeding with full dataset.")
            return data

    elif subset_option.startswith("random"):
        sample_size = int(input("Enter the number of random samples: ").strip())
        return data.sample(n=sample_size)

    elif subset_option == "no subset":
        return data

    else:
        print("Invalid option. Proceeding with full dataset.")
        return data

def display_header(query, doc_id):
    print("\n" + "="*80)
    print("Welcome to rerank feedback interface!")
    print("="*80)
    print(f"\nCurrently evaluating document ID: {doc_id}")
    print(f"Query being evaluated: \"{query}\"")
    print("-"*80)

def generate_query(base_query, iso_code):
    """Generate a query using the ISO code."""
    return base_query.format(iso_code=iso_code)

def load_and_filter_data(file_path):
    """Load data and apply initial filtering."""
    print(f"Loading documents from {file_path}")
    df = pd.read_csv(file_path)
    
    # Ask about self-mentions
    include_self_mentions = input("Do you want to include self-mentions? (yes/no): ").strip().lower()
    if include_self_mentions == "no":
        if 'iso' in df.columns and 'ISO_Code' in df.columns:  # Fixed case sensitivity
            df = df[df['iso'] != df['ISO_Code']]  # Fixed column names
            print("Self-mentions filtered out.")
        else:
            print("Columns 'iso' and 'ISO_Code' not found. Proceeding with full dataset.")
    
    # Apply subsetting if requested
    df = subset_data(df)
    return df

def process_document(co, base_query, doc_chunks, results_dict, full_data):
    # Get the ISO code from the target ISO_Code
    iso_code = doc_chunks[0].get('ISO_Code', '')  # Fixed column name
    query = generate_query(base_query, iso_code)
    
    chunk_texts = [chunk['text'] for chunk in doc_chunks]
    
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=chunk_texts,
        top_n=len(chunk_texts)
    )
    
    for i, result in enumerate(response.results):
        chunk = doc_chunks[result.index]
        print(f"\nChunk {i+1}/{len(response.results)}")
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Target Country (ISO Code): {iso_code}")
        print(f"Source Country (ISO Code): {chunk.get('iso', '')}")  # Added source country
        print(f"Relevance Score: {result.relevance_score:.4f}")
        print(f"Text: {chunk['text']}")
        print(f"\nQuery: \"{query}\"")
        
        while True:
            try:
                annotation = input("\nDoes this address the request? (1 for yes, 0 for no): ").strip()
                if annotation in ['0', '1']:
                    break
                print("Please enter either 0 or 1")
            except KeyError:
                print("Invalid input. Please enter 0 or 1.")
        
        # Update the full_data with annotation
        chunk_idx = full_data.index[full_data['chunk_id'] == chunk['chunk_id']].tolist()[0]
        full_data.at[chunk_idx, 'human_annotation'] = int(annotation)
        full_data.at[chunk_idx, 'relevance_score'] = result.relevance_score
        
        if i < len(response.results) - 1:
            while True:
                print("\nOptions:")
                print("1) Continue to next chunk")
                print("2) Exit and mark remaining as NaN")
                print("3) Exit and mark remaining as 0")
                print("4) Mark all remaining as 0 and proceed to next doc_id")
                print("5) Exit and save annotations")
                choice = input("Choice: ").strip()
                
                if choice in ['1', '2', '3', '4', '5']:
                    break
                print("Please enter a number between 1 and 5")
            
            if choice == '2':  # Exit with NaN
                for remaining in response.results[i+1:]:
                    remaining_chunk = doc_chunks[remaining.index]
                    chunk_idx = full_data.index[full_data['chunk_id'] == remaining_chunk['chunk_id']].tolist()[0]
                    full_data.at[chunk_idx, 'human_annotation'] = "NaN"
                    full_data.at[chunk_idx, 'relevance_score'] = remaining.relevance_score
                return "exit"
            elif choice in ['3', '4']:  # Exit with 0 or mark remaining as 0 and continue
                for remaining in response.results[i+1:]:
                    remaining_chunk = doc_chunks[remaining.index]
                    chunk_idx = full_data.index[full_data['chunk_id'] == remaining_chunk['chunk_id']].tolist()[0]
                    full_data.at[chunk_idx, 'human_annotation'] = 0
                    full_data.at[chunk_idx, 'relevance_score'] = remaining.relevance_score
                return "next" if choice == '4' else "exit"
            elif choice == '5':  # Exit and save
                return "save"
    
    return "next"

def main():
    co = get_cohere_client()
    
    # Load documents
    default_file = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\ungdc_chunk_model-v5_EntityContext.csv"
    file_path = input(f"Enter the path to the input file (or press Enter to use default): ").strip()
    if not file_path:
        file_path = default_file
    
    # Load and filter full dataset
    original_data = load_and_filter_data(file_path)
    
    # Add new columns for annotations and relevance scores
    original_data['human_annotation'] = None
    original_data['relevance_score'] = None
    
    # Group documents
    doc_groups = defaultdict(list)
    for _, row in original_data.iterrows():
        doc_groups[row['doc_id']].append({
            'chunk_id': row['chunk_id'],
            'text': row['text'],
            'ISO_Code': row['ISO_Code'],  # Fixed column name
            'iso': row['iso']  # Added source ISO
        })
    
    base_query = "Does this text overtly praise {iso_code}?"
    
    for doc_id, doc_chunks in doc_groups.items():
        display_header(base_query.format(iso_code=doc_chunks[0]['ISO_Code']), doc_id)  # Fixed column name
        
        status = process_document(co, base_query, doc_chunks, None, original_data)
        
        # Save after each document
        output_file = 'rerank_annotations.jsonl'
        original_data.to_json(output_file, orient='records', lines=True)
        print(f"\nResults saved to {output_file}")
        
        if status == "exit" or status == "save":
            break
        elif status == "next":
            continue
        
        if doc_id != list(doc_groups.keys())[-1]:
            proceed = input("\nWould you like to proceed to the next document? (y/n): ").strip().lower()
            if proceed != 'y':
                break

if __name__ == "__main__":
    main()