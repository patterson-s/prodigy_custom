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

def save_results(results: list, output_folder: str) -> None:
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "results.jsonl")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Results saved to {output_file}")

def load_previous_results(output_folder: str) -> list:
    results = []
    output_file = os.path.join(output_folder, "results.jsonl")
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                results.append(json.loads(line.strip()))
        print(f"Loaded previous results from {output_file}")
    return results

def main() -> None:
    # Initialize Cohere client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    co_client = cohere.ClientV2(api_key)

    # Get output folder
    output_folder = input("Enter the output folder path: ").strip()
    
    # Load previous results if available
    results = load_previous_results(output_folder)
    completed_rows = {result["Row"] for result in results}

    # Load and subset dataset
    data = pd.read_csv(input("Enter the path to the input file: ").strip())
    
    for index, row in data.iterrows():
        if index + 1 in completed_rows:
            continue

        print(f"\nProcessing row {index + 1}...")
        
        # Analyze text
        analysis = analyze_social_facts(
            co_client,
            row['source'],
            row['target'],
            str(row['year']),
            row['text']
        )
        
        print("\nAnalysis Output:")
        print(analysis)
        
        # Collect evaluation
        while True:
            try:
                adequate = int(input("\nIs this output adequate? (1 for yes, 0 for no): ").strip())
                if adequate in [0, 1]:
                    break
                print("Please enter 0 or 1")
            except ValueError:
                print("Invalid input. Please enter 0 or 1")
                
        while True:
            try:
                likert = int(input("Rate quality on scale of 1-5 (5 being perfect): ").strip())
                if 1 <= likert <= 5:
                    break
                print("Please enter a number between 1 and 5")
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 5")
                
        note = input("Enter any notes (press Enter if none): ").strip()

        # Save results
        result = {
            "Row": index + 1,
            **row.to_dict(),
            "Analysis": analysis,
            "Adequate": adequate,
            "Likert": likert,
            "Note": note
        }
        results.append(result)
        save_results(results, output_folder)

        # Check if user wants to continue
        proceed = input("\nContinue to next row? (yes/no): ").strip().lower()
        if proceed != "yes":
            break

    print("Analysis complete!")

if __name__ == "__main__":
    main()