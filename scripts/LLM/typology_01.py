from typing import Dict, List
import json
import cohere
import os
import datetime

BATCH_SIZE = 20

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

def process_batches(data: List[Dict], batch_size: int = 20) -> List[List[Dict]]:
    """Split data into batches, allowing final batch to be smaller"""
    batches = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batches.append(batch)
    return batches

def analyze_diplomatic_batch(
    client: cohere.ClientV2,
    batch: List[Dict],
    batch_num: int,
    output_dir: str,
    temperature: float = 0.7
) -> List[Dict]:
    """Process a batch through Cohere API and parse results"""
    
    # Construct system and user messages
    system_message = {
        "role": "system",
        "content": (
            "You are an expert analyst specializing in diplomatic discourse analysis. "
            "Your task is to analyze summaries of diplomatic speeches to classify rhetorical acts and identify key themes. "
            "Format your response in markdown with clear headers for Row, Source, Target, Text, Proposed Label, and Proposed Tags for each summary."
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
        
        # Save raw response
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = os.path.join(output_dir, f"raw_response_batch_{batch_num}_{timestamp}.txt")
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        # Parse the markdown response
        results = parse_markdown_response(combined_content, batch_num)
        
        return results
    except Exception as e:
        print(f"Error processing batch: {e}")
        raise

import re

def parse_markdown_response(response: str, batch_num: int) -> List[Dict]:
    """Parse markdown-formatted response into structured data"""
    results = []
    
    # Split into summaries
    summaries = response.split("**Summary")[1:]  # Skip the header
    
    for summary in summaries:
        try:
            # Extract row number
            row_match = re.search(r"\*\*Row:\*\* (\d+)", summary)
            if not row_match:
                continue
            row = int(row_match.group(1))
            
            # Extract source
            source_match = re.search(r"\*\*Source:\*\* ([A-Z]{3})", summary)
            if not source_match:
                continue
            source = source_match.group(1)
            
            # Extract target
            target_match = re.search(r"\*\*Target:\*\* ([A-Z]{3})", summary)
            if not target_match:
                continue
            target = target_match.group(1)
            
            # Extract text
            text_match = re.search(r"\*\*Text:\*\* ([^\n]+)", summary)
            if not text_match:
                continue
            text = text_match.group(1)
            
            # Extract label - handle both formats that might appear
            label_match = re.search(r"\*\*Proposed Label:\*\* \*\*([^\n]+)\*\*|\*\*Proposed Label:\*\* ([^\n]+)", summary)
            if not label_match:
                continue
            label = label_match.group(1) if label_match.group(1) else label_match.group(2)
            
            # Extract tags
            tags_match = re.search(r"\*\*Proposed Tags:\*\* ([^\n]+)", summary)
            if not tags_match:
                continue
            # Convert comma or semicolon separated tags into semicolon-separated format
            tags = tags_match.group(1)
            # Split on either comma or semicolon, strip whitespace, and rejoin with semicolons
            tags = ';'.join(t.strip() for t in re.split('[,;]', tags))
            
            result = {
                "Row": row,
                "source": source,
                "target": target,
                "text": text,
                "proposed_label": label.strip(),
                "proposed_tags": tags,
                "batch_num": batch_num
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error parsing summary in batch {batch_num}: {e}")
            continue
    
    if not results:
        print(f"Warning: No valid summaries found in batch {batch_num}")
        print("Raw response excerpt:", response[:200] + "...")
    
    return results

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
    output_dir = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\typology_01"
    os.makedirs(output_dir, exist_ok=True)

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
        all_results = []
        for i, batch in enumerate(batches):
            batch_num = i + 1
            print(f"Processing batch {batch_num}/{len(batches)}...")
            
            try:
                # Process batch and get parsed results
                results = analyze_diplomatic_batch(client, batch, batch_num, output_dir)
                all_results.extend(results)
                
                print(f"Processed {len(results)} items from batch {batch_num}")
                
            except Exception as e:
                print(f"Error in batch {batch_num}: {e}")
                continue  # Continue with next batch instead of failing
        
        # Save final results
        if all_results:
            save_results(all_results, output_dir)
            print(f"Analysis complete. Processed {len(all_results)} items total.")
        else:
            print("Warning: No results were successfully processed.")

    except Exception as e:
        print(f"Error in processing: {e}")
        raise

if __name__ == "__main__":
    main()
