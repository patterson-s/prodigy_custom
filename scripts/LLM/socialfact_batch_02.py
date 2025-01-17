import os
import cohere
import pandas as pd
import json
from typing import Dict, Any

def analyze_social_facts(co_client: cohere.ClientV2, 
                        source: str, 
                        target: str, 
                        year: str, 
                        text: str) -> str:
    system_message = {
        "role": "system",
        "content": "You are a Constructivist analyst of International Relations, expert in observing and interpreting diplomatic language."
    }
    
    prompt_template = """# Analysis of UNGA Speech: {SOURCE} on {TARGET}, {YEAR}

## Theoretical Framework

You are a Constructivist analyst of International Relations. Your job is to observe and interpret how world politics are intersubjectively constructed. You are an expert observer and interpreter of diplomatic language.

When diplomats speak in public, they are not simply describing the world as it truly exists. Rather, they portray an image of the world in which their country is a protagonist. One key way they do this is by attempting to create social facts about other states.

Social facts are facts that do not exist in the physical world but exist because people collectively agree that they exist and often require institutions to maintain their existence. For example, money is a social fact - paper bills have value because we collectively agree they do and because institutions maintain this agreement. In international relations, many important facts are social facts - like whether a state is considered "peaceful," "aggressive," "democratic," or "responsible."

When diplomats speak at the UN General Assembly, they are participating in the creation, maintenance, and contestation of social facts about other states. They do this through acts of essentialization - statements that attribute essential characteristics or qualities to other states.

Text to analyze:
{TEXT}

Follow this analytical process:

1. Social Facts Construction:
   - What social facts is {SOURCE} trying to create or reinforce about {TARGET}?
   - Are they contesting existing social facts about {TARGET}?
   - What characteristics or qualities are being essentialized?

2. Methods and Evidence:
   - How does {SOURCE} attempt to establish these social facts?
   - What kinds of claims, evidence, or rhetorical strategies are employed?
   - How do they use historical references, current events, or future projections?
   - What institutional or normative frameworks do they invoke?

3. Contextual Analysis:
   - How does this characterization relate to broader patterns in international relations?
   - What is the historical context of this characterization?
   - How does it connect to existing hierarchies or status relationships?
   - What are the potential implications for international order?

4. Linguistic and Rhetorical Analysis:
   - What diplomatic conventions or indirect language is being used?
   - How does the choice of words contribute to the essentialization?
   - What is left unsaid but implied?
   - How does the statement operate on multiple levels or speak to different audiences?

Provide your analysis in two parts:

1. Reasoning: A step-by-step analysis following the above process.

2. Output: In a single clear sentence, describe:
   - How {TARGET} is being characterized or portrayed
   - The methods or evidence used to create this portrayal"""

    user_message = prompt_template.format(
        SOURCE=source,
        TARGET=target,
        YEAR=year,
        TEXT=text
    )

    try:
        response = co_client.chat(
            model="command-r7b-12-2024",
            messages=[
                system_message,
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=4000
        )
        return ''.join(segment.text for segment in response.message.content).strip()
    except Exception as e:
        return f"Error: {str(e)}"

def save_batch_results(results: list, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "batch_results.jsonl")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"\nResults saved to {output_file}")

def main() -> None:
    # Initialize Cohere client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    co_client = cohere.ClientV2(api_key)

    # Get input file and output folder
    input_file = input("Enter the path to the input file: ").strip()
    output_folder = input("Enter the output folder path: ").strip()
    
    # Load dataset
    print("\nLoading dataset...")
    data = pd.read_csv(input_file)
    
    # Optional year filtering
    year_filter = input("\nEnter year to filter by (press Enter for full dataset): ").strip()
    if year_filter:
        try:
            year = int(year_filter)
            data = data[data['year'] == year]
            print(f"Dataset filtered to year {year}")
        except ValueError:
            print("Invalid year format. Processing full dataset.")
    
    print(f"\nTotal rows to process: {len(data)}")
    
    # Process data
    results = []
    print("\nStarting processing...")
    
    total_rows = len(data)
    for count, (_, row) in enumerate(data.iterrows(), 1):
        analysis = analyze_social_facts(
            co_client,
            row['source'],
            row['target'],
            str(row['year']),
            row['text']
        )
        
        result = {
            "Row": count,
            **row.to_dict(),
            "Analysis": analysis
        }
        results.append(result)
        
        # Update progress and save every 50 rows
        if count % 50 == 0:
            print(f"Progress: {count}/{total_rows} rows processed")
            save_batch_results(results, output_folder)

    # Final save
    save_batch_results(results, output_folder)
    print("\nBatch analysis complete!")

if __name__ == "__main__":
    main()