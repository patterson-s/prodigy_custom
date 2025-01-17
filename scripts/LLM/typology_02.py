from typing import Dict, List, Optional
import json
import cohere
import os
import datetime
import re

# The core prompt for typology development
TYPOLOGY_PROMPT = '''You are an expert analyst specializing in diplomatic discourse analysis and international relations theory. Your expertise includes deep knowledge of how states construct relationships and position themselves through language in international forums. You have extensive experience analyzing diplomatic speech at the United Nations and understand how states use language to establish, maintain, and transform their relationships with other states.

Your task is to analyze a batch of diplomatic speech summaries to develop a systematic typology of diplomatic discourse. This analysis is part of an abductive approach where patterns emerge naturally from close reading of the texts. Rather than classifying individual speeches, your goal is to identify and articulate the fundamental types of rhetorical acts and content domains that characterize diplomatic discourse.

Key Theoretical Framework:
- Focus on how texts construct relationships and status distinctions between states
- Analyze both fundamental rhetorical acts (which will become labels) and content domains (which will become tags)
- Consider how states position themselves relative to others in the international system
- Examine how language choices reflect and construct status relationships
- Identify patterns in how states assert agency and attribute responsibility

Analysis Process:
1. Initial Reading
- Read all summaries carefully to get a holistic view
- Note recurring patterns in how states relate to each other
- Identify common themes and diplomatic moves

2. Pattern Identification
- What fundamental diplomatic moves appear repeatedly?
- How do states consistently position themselves?
- What relationship types are being constructed?
- What content domains recur across different speeches?

3. Label Development
- Identify distinct categories of rhetorical acts
- Ensure categories are mutually exclusive
- Focus on relational aspects of diplomatic speech
- Consider both explicit and implicit relationship construction
- Create clear, descriptive names for each category

4. Tag Development
- Identify recurring content domains
- Look for thematic patterns across speeches
- Consider both subject matter and diplomatic context
- Create standardized terminology for similar concepts

5. Typology Construction
- Organize labels into a coherent system
- Ensure coverage of observed diplomatic moves
- Create clear definitions for each category
- Note relationships between different types

6. Documentation
- Explain your reasoning process
- Support category choices with examples
- Define boundaries between categories
- Describe how the typology captures diplomatic dynamics

Your final output should be structured as a JSON object containing:
{
  "labels": [
    {
      "name": "label_name",
      "description": "Clear definition of the diplomatic move",
      "examples": ["Example diplomatic moves that fit this category"]
    }
  ],
  "tags": [
    {
      "name": "tag_name",
      "description": "Definition of this content domain",
      "examples": ["Example topics or themes in this domain"]
    }
  ]
}

Before providing your typology, explain your analysis process and how you identified the patterns that led to your categorization scheme. Show your reasoning for how you determined the fundamental types of diplomatic moves and content domains.'''

def setup_cohere() -> cohere.ClientV2:
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
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

def extract_json_from_response(response: str) -> Optional[Dict]:
    """Extract JSON object from the response text"""
    try:
        # Find JSON-like structure between curly braces
        json_match = re.search(r'\{[^{]*"labels"[^}]*"tags"[^}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        
        print("No JSON found in response. Saving raw response for inspection.")
        return None
    except Exception as e:
        print(f"Error extracting JSON: {e}")
        return None

def analyze_batch(
    client: cohere.ClientV2,
    batch: List[Dict],
    batch_num: int,
    output_dir: str,
    temperature: float = 0.7
) -> Optional[Dict]:
    """Process a batch through Cohere API"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construct messages
    system_message = {
        "role": "system",
        "content": "You are an expert analyst specializing in diplomatic discourse analysis."
    }
    user_message = {
        "role": "user",
        "content": TYPOLOGY_PROMPT + format_batch_for_prompt(batch)
    }

    try:
        # Get API response
        response = client.chat(
            model="command-r7b-12-2024",
            messages=[system_message, user_message],
            temperature=temperature
        )

        # Combine response content
        combined_content = ''.join(item.text for item in response.message.content if item.type == "text").strip()
        
        # Save raw response
        raw_filename = f"raw_response_batch_{batch_num}_{timestamp}.txt"
        raw_path = os.path.join(output_dir, raw_filename)
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(combined_content)
        
        # Extract and return JSON
        return extract_json_from_response(combined_content)
        
    except Exception as e:
        print(f"Error processing batch {batch_num}: {e}")
        return None

def main():
    # Initialize Cohere client
    client = setup_cohere()

    # File paths
    input_file = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\agency_eval_09\agency_eval_09_goldstandard_filtered.jsonl"
    output_dir = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\typology_02"
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
        
        # Process each batch and collect results
        all_typologies = []
        for i, batch in enumerate(batches):
            batch_num = i + 1
            print(f"\nProcessing batch {batch_num}/{len(batches)}...")
            
            typology = analyze_batch(client, batch, batch_num, output_dir)
            if typology:
                typology['batch_num'] = batch_num
                all_typologies.append(typology)
                print(f"Successfully extracted typology from batch {batch_num}")
            else:
                print(f"Failed to extract typology from batch {batch_num}")

        # Save final results
        if all_typologies:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(output_dir, f"typologies_{timestamp}.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_typologies, f, indent=2)
            print(f"\nSaved {len(all_typologies)} typologies to {results_file}")
        else:
            print("\nWarning: No typologies were successfully processed")

    except Exception as e:
        print(f"Error in processing: {e}")
        raise

if __name__ == "__main__":
    main()