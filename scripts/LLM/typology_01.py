from typing import Dict, List
import json
import cohere
import os
import datetime

BATCH_SIZE = 5

PROMPT_1_1 = """You are an expert analyst specializing in diplomatic discourse analysis and international relations theory. Your expertise includes deep knowledge of how states construct relationships and position themselves through language. You have extensive experience analyzing diplomatic speech at the United Nations and understand how states use language to establish, maintain, and transform their relationships with other states.

Your task is to analyze a batch of diplomatic speech summaries to identify and categorize how states characterize their relationships with other states. This analysis is part of an abductive approach to developing a typology of diplomatic discourse by allowing patterns to emerge naturally from the data.

Key Theoretical Framework:
- Focus on how texts construct relationships and distinctions between the $SOURCE and $TARGET states
- Analyze both the fundamental rhetorical acts (labels) and content domains (tags)
- Consider how states position themselves relative to others in the international system
- Examine how language choices reflect and construct status relationships

For each summary in the batch, you will:
1. Read and analyze the text carefully
2. Identify the fundamental rhetorical act being performed (this will become the label)
3. Identify relevant content domains (these will become tags)
4. Generate structured output in the required JSON format

Requirements for Labels:
- Each summary must receive exactly one label
- Labels must capture the fundamental rhetorical act being performed
- Labels must be mutually exclusive
- No label stacking - if multiple elements combine, create a distinct new label
- Focus on the relational aspect of diplomatic speech
- Maintain consistency across the batch

Requirements for Tags:
- Multiple tags are allowed and expected
- Use semicolons to separate multiple tags
- Tags should capture content domains and themes
- Maintain consistent terminology across the batch
- Use noun phrases when possible

Required Output Format:
{
  "Row": [id],
  [original_fields],
  "proposed_label": "single_label_here",
  "proposed_tags": "tag1;tag2;tag3"
}

Analysis Process:
1. Initial Assessment
- Read the summary carefully
- Identify key themes and relationships

2. Rhetorical Analysis
- What is the fundamental diplomatic move being made?
- How does the speaking state position itself relative to others?
- What relationship is being constructed or maintained?

3. Content Analysis
- What domains or topics are being addressed?
- What themes emerge from the language used?
- How do content choices support the rhetorical act?

4. Label Generation
- Identify the core rhetorical act
- Ensure it captures the fundamental diplomatic move
- Verify it is distinct from other labels in the batch
- Confirm it focuses on the relational aspect
- Express the label as a form of action

5. Tag Assignment
- List all relevant content domains
- Standardize terminology
- Combine with semicolons
- Verify consistency with other tags in batch

6. Output Validation
- Verify JSON format
- Check label meets criteria
- Confirm tags are properly formatted
- Ensure all original fields are preserved

Before providing your analysis, explain your reasoning process for each step. Show how you identified the rhetorical act, why you chose that specific label, and how you determined the relevant tags. This explanation helps ensure systematic, theoretically-grounded analysis.

Remember:
- Focus on emerging patterns within the current batch
- Maintain consistent terminology
- Keep the theoretical framework in mind
- Document your reasoning clearly"""

def setup_cohere() -> cohere.ClientV2:
    """Initialize the Cohere client with API key"""
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("Set your COHERE_API_KEY environment variable first!")
    return cohere.ClientV2(api_key)

def format_batch_for_prompt(batch: List[Dict]) -> str:
    """Format a batch of summaries for the prompt"""
    formatted_text = "\n\nBatch of Diplomatic Summaries:\n"
    for item in batch:
        formatted_text += f"\nSummary {item['row']}:\n"
        formatted_text += f"Source: {item['source']}\n"
        formatted_text += f"Target: {item['target']}\n"
        formatted_text += f"Text: {item['summary']}\n"
        formatted_text += "---"
    return formatted_text

def process_batches(data: List[Dict]) -> List[List[Dict]]:
    """Split data into batches of BATCH_SIZE"""
    return [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]

def analyze_diplomatic_batch(
    client: cohere.ClientV2,
    batch: List[Dict],
    temperature: float = 0.7
) -> str:
    """Process a batch through Cohere API"""
    if len(batch) != BATCH_SIZE:
        raise ValueError(f"Batch must contain exactly {BATCH_SIZE} items")

    # Construct system and user messages
    system_message = {
        "role": "system",
        "content": (
            "You are an expert analyst specializing in diplomatic discourse analysis. "
            "Your task is to analyze summaries of diplomatic speeches to classify rhetorical acts and identify key themes."
        )
    }
    user_message = {
        "role": "user",
        "content": PROMPT_1_1 + format_batch_for_prompt(batch)
    }

    try:
        response = client.chat(
            model="command-r7b-12-2024",
            messages=[system_message, user_message],
            temperature=temperature
        )

        # Combine the content from response.message.content
        combined_content = ''.join(item.text for item in response.message.content if item.type == "text").strip()
        return combined_content
    except Exception as e:
        print(f"Error processing batch: {e}")
        raise


import re

def parse_api_response(response: str) -> List[Dict]:
    """Extract and parse JSON blocks from the API response."""
    try:
        # Use regex to extract all JSON blocks
        json_blocks = re.findall(r"\{.*?\}", response, re.DOTALL)
        parsed_results = [json.loads(block) for block in json_blocks]

        if not parsed_results:
            # Log the full response for debugging
            with open("failed_responses.log", "a") as log_file:
                log_file.write(f"Raw response:\n{response}\n\n")
            raise ValueError("No valid JSON found in the response.")

        return parsed_results
    except json.JSONDecodeError as e:
        print(f"Error parsing API response: {e}")
        print("Raw response content:", response)
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        print("Raw response content:", response)
        raise


def save_results(results: List[Dict], output_dir: str) -> None:
    """Save the analysis results to a JSONL file."""
    try:
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"diplomatic_analysis_{timestamp}.jsonl")
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
        raise


def main():
    # Initialize Cohere client
    client = setup_cohere()

    # Specify input and output file paths
    input_file = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\agency_eval_09\agency_eval_09_goldstandard_filtered.jsonl"
    output_dir = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\typology_01"  # Specify output directory

    try:
        # Read input data
        data = []
        with open(input_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                if 'Filtered Prompt Output' in entry:
                    data.append({
                        'row': entry['Row'],
                        'source': entry['source'],
                        'target': entry['target'],
                        'summary': entry['Filtered Prompt Output']
                    })
        if not data:
            raise ValueError("No valid data found in input file")

        print(f"Loaded {len(data)} entries from {input_file}")

        # Process in batches
        batches = process_batches(data)

        # Process each batch
        results = []
        for i, batch in enumerate(batches):
            print(f"Processing batch {i+1}/{len(batches)}...")
            try:
                response = analyze_diplomatic_batch(client, batch)
                parsed_results = parse_api_response(response)
                results.extend(parsed_results)
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
        
        # Save results
        save_results(results, output_dir)
        print("Analysis complete.")

    except Exception as e:
        print(f"Error in processing: {e}")
        raise

if __name__ == "__main__":
    main()
