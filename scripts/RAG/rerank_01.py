import cohere
import os
import json
from datetime import datetime
from pathlib import Path

def get_cohere_client():
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    return cohere.ClientV2(api_key)

def save_to_jsonl(data, filename):
    with open(filename, 'a') as f:
        json.dump(data, f)
        f.write('\n')

def display_header(query):
    print("\n" + "="*80)
    print("Welcome to rerank feedback interface!")
    print("="*80)
    print(f"\nQuery being evaluated: \"{query}\"")
    print("-"*80)

def main():
    # Initialize Cohere client
    co = get_cohere_client()
    
    # Sample documents (you can modify these or load from a file)
    docs = [
        "Carson City is the capital city of the American state of Nevada.",
        "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
        "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
        "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
        "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
    ]
    
    query = "What is the capital of the United States?"
    
    # Display welcome message and query
    display_header(query)
    
    # Get rerank results
    response = co.rerank(
        model="rerank-v3.5",
        query=query,
        documents=docs,
        top_n=len(docs)  # Get all results to handle all documents
    )
    
    # Initialize results dictionary
    results = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'annotations': []
    }
    
    # Process each result
    for i, result in enumerate(response.results):
        print(f"\nDocument {i+1}/{len(response.results)}")
        print(f"Relevance Score: {result.relevance_score:.4f}")
        print(f"Text: {docs[result.index]}")
        print(f"\nQuery: \"{query}\"")  # Repeat query for reference
        
        while True:
            try:
                annotation = input("\nDoes this address the request? (1 for yes, 0 for no): ").strip()
                if annotation in ['0', '1']:
                    break
                print("Please enter either 0 or 1")
            except KeyError:
                print("Invalid input. Please enter 0 or 1.")
        
        # Add annotation to results
        results['annotations'].append({
            'document_index': result.index,
            'document_text': docs[result.index],
            'relevance_score': result.relevance_score,
            'human_annotation': int(annotation)
        })
        
        if i < len(response.results) - 1:  # If not the last document
            while True:
                choice = input("\nWould you like to:\n1) Continue to next document\n2) Exit and mark remaining as NaN\n3) Exit and mark remaining as 0\nChoice: ").strip()
                
                if choice in ['1', '2', '3']:
                    break
                print("Please enter 1, 2, or 3")
            
            if choice == '2':  # Exit with NaN
                for remaining in response.results[i+1:]:
                    results['annotations'].append({
                        'document_index': remaining.index,
                        'document_text': docs[remaining.index],
                        'relevance_score': remaining.relevance_score,
                        'human_annotation': "NaN"
                    })
                break
            elif choice == '3':  # Exit with 0
                for remaining in response.results[i+1:]:
                    results['annotations'].append({
                        'document_index': remaining.index,
                        'document_text': docs[remaining.index],
                        'relevance_score': remaining.relevance_score,
                        'human_annotation': 0
                    })
                break
    
    # Save results
    output_file = 'rerank_annotations.jsonl'
    save_to_jsonl(results, output_file)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()