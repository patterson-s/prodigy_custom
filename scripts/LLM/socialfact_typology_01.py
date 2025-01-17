import os
import cohere
import pandas as pd
import json
from typing import List, Optional, Tuple

def load_social_facts(file_path: str, target: str, year: int) -> List[str]:
    """
    Filters the dataset for all 'extracted_output' values where:
    - target matches the specified state
    - source != target
    - year matches the specified year
    """
    data = pd.read_json(file_path, lines=True)
    filtered_data = data[(data['target'] == target) & 
                         (data['source'] != target) & 
                         (data['year'] == year)]
    return filtered_data['extracted_output'].tolist()

def generate_typology(co_client: cohere.ClientV2, 
                     target: str, 
                     year: int, 
                     social_facts: List[str]) -> dict:
    system_message = {
        "role": "system",
        "content": "You are a Constructivist analyst of International Relations, expert in synthesizing and abstracting patterns in diplomatic language."
    }

    prompt_template = """# Typology Development for Social Facts about {TARGET}, {YEAR}

IMPORTANT: Your response MUST end with a clearly marked "Output:" section containing a structured typology.

## Theoretical Framework

You are a Constructivist analyst of International Relations. Your task is to synthesize patterns and create a typology of social facts about the target state, {TARGET}, for the year {YEAR}. Social facts are intersubjectively constructed understandings that shape perceptions and norms in international relations. They are created and contested through diplomatic language, essentializing specific characteristics or roles for states.

For this task, you are provided with a set of social facts about {TARGET}, created by other states ({TARGET} does not speak about itself in this data). Your goal is to:

1. Analyze the dataset to identify patterns, themes, and shared characteristics among the social facts.
2. Create a typology that organizes these social facts into 1–5 distinct categories, based on shared themes or functions in diplomatic discourse.
3. Justify the categories with reasoning and examples from the dataset.

## Social Facts Dataset

The following are the social facts for {TARGET} in {YEAR}:

{SOCIAL_FACTS}

## Analysis Requirements

Provide your analysis in exactly TWO parts:

PART 1 - Reasoning:
Follow this analytical process step-by-step:

1. **Pattern Identification**:
   - Reflect on the social facts as a whole.
   - Identify recurring themes, shared characteristics, or contrasting portrayals among the social facts.
   - Consider tone (e.g., critical, sympathetic), subject (e.g., government, people, or role in international order), and rhetorical strategies.

2. **Clustering and Abstraction**:
   - Group the social facts into categories based on shared themes or characteristics.
   - Determine if there are overlaps, outliers, or ambiguities in the dataset that affect grouping.

3. **Reasoning and Refinement**:
   - Justify the categories you create, explaining the criteria used for grouping.
   - Ensure that the categories are distinct and capture the full range of social facts in the dataset.

PART 2 - Required Output:
You MUST end your analysis with:
1. A clearly structured typology of 1–5 categories. Each category must include:
   - A descriptive label.
   - A concise explanation of the category.
   - 1–3 example social facts from the dataset that fit within this category.
2. A concluding statement summarizing the typology and its significance.

REMINDER: Your analysis MUST conclude with the "Output:" section containing the typology in a structured JSON-like format.
"""

    # Join the social facts into a single string for the prompt
    social_facts_text = "\n".join([f"- {fact}" for fact in social_facts])

    user_message = prompt_template.format(
        TARGET=target,
        YEAR=year,
        SOCIAL_FACTS=social_facts_text
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
        response_text = ''.join(segment.text for segment in response.message.content).strip()
        # Extract typology section from the response
        if "Output:" in response_text:
            output_start = response_text.find("Output:") + len("Output:")
            typology_section = response_text[output_start:].strip()
            return {"reasoning": response_text[:output_start].strip(), "typology": typology_section}
        else:
            return {"reasoning": response_text, "typology": "Error: Typology not found in response."}
    except Exception as e:
        return {"reasoning": f"Error: {str(e)}", "typology": None}

def save_typology_results(results: List[dict], output_file: str) -> None:
    """Saves the typology results to a JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

def main() -> None:
    # Initialize Cohere client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    co_client = cohere.ClientV2(api_key)

    # Define dataset path and output folder
    input_file = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\analysis\socialfact_batch_03_1946-7_filtered.jsonl"
    output_file = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\analysis\typology_results.jsonl"

    # Load dataset
    print("\nLoading dataset...")
    data = pd.read_json(input_file, lines=True)

    # Get unique target-year pairs
    unique_targets = data[['target', 'year']].drop_duplicates()

    # Initialize results
    all_results = []

    # Process each target-year pair
    for _, row in unique_targets.iterrows():
        target = row['target']
        year = row['year']

        print(f"\nProcessing target: {target}, year: {year}...")

        # Load social facts for the specific target and year
        social_facts = load_social_facts(input_file, target, year)

        if not social_facts:
            print(f"No social facts found for target {target} in year {year}.")
            continue

        # Generate typology
        typology_result = generate_typology(co_client, target, year, social_facts)

        # Append results
        all_results.append({"target": target, "year": year, "typology": typology_result})

    # Save all results
    print("\nSaving all results...")
    save_typology_results(all_results, output_file)

    print(f"\nTypology generation complete! Results saved to {output_file}")

if __name__ == "__main__":
    main()
