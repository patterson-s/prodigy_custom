import os
import cohere
import pandas as pd
import json
import random

# Initialize Cohere client
api_key = os.getenv('COHERE_API_KEY')
if not api_key:
    raise ValueError("COHERE_API_KEY environment variable not found")
co_client = cohere.ClientV2(api_key)

def analyze_with_prompts(co_client, target, text):
    """Run all prompts and return their outputs."""
    system_message = {
        "role": "system",
        "content": (
            "You are an advanced language model trained to analyze diplomatic speech. "
            "Your task is to process the provided text based on the user's instructions."
        )
    }

    prompts = [
        {"name": "Prompt 1", "message": {
            "role": "user", "content": (
                f"Rewrite the context in which the target state is mentioned into a single sentence.\n\n"
                f"Target: {target}\n"
                f"Text: {text}\n\n"
                f"Instructions:\n"
                f"1. Begin with the target state.\n"
                f"2. Use as much of the original language as possible.\n"
                f"3. Ensure grammatical correctness by addressing issues such as conjugations or sentence structure.\n"
                f"4. Preserve the essence and meaning of the original text."
            )
        }},
        {"name": "Prompt 2", "message": {
            "role": "user", "content": (
                f"Summarize the mention of the target state in the text using original language.\n\n"
                f"Target: {target}\n"
                f"Text: {text}\n\n"
                f"Instructions:\n"
                f"1. Start with the target state.\n"
                f"2. Use concise, original wording where possible.\n"
                f"3. Maintain grammatical accuracy.\n"
                f"4. Ensure the meaning reflects the original context."
            )
        }},
        {"name": "Prompt 3", "message": {
            "role": "user", "content": (
                f"Analyze the tone and sentiment towards the target state in the provided text.\n\n"
                f"Target: {target}\n"
                f"Text: {text}\n\n"
                f"Instructions:\n"
                f"1. Begin with the target state.\n"
                f"2. Clearly state whether the tone is positive, negative, or neutral.\n"
                f"3. Provide evidence from the text to support your assessment."
            )
        }},
        {"name": "Prompt 4", "message": {
            "role": "user", "content": (
                f"Extract and list all explicit mentions of the target state and their context from the provided text.\n\n"
                f"Target: {target}\n"
                f"Text: {text}\n\n"
                f"Instructions:\n"
                f"1. Provide each mention as a separate item.\n"
                f"2. Include sufficient context for each mention.\n"
                f"3. Ensure that the output is clear and easy to interpret."
            )
        }}
    ]

    results = {}
    for prompt in prompts:
        try:
            response = co_client.chat(
                model="command-r7b-12-2024",
                messages=[system_message, prompt["message"]]
            )
            # Correctly concatenate response segments
            results[prompt["name"]] = ''.join(segment.text for segment in response.message.content).strip()
        except Exception as e:
            results[prompt["name"]] = f"Error: {str(e)}"

    return results


def subset_data(data):
    """Subset the dataset based on user input."""
    print("Would you like to subset the data? Options: 'year', 'random x', 'no subset'")
    subset_option = input("Enter your choice: ").strip().lower()

    # Normalize all column names to lowercase
    data.columns = [col.lower() for col in data.columns]

    if subset_option == "year":
        if 'year' not in data.columns:
            print("Column 'year' not found in dataset.")
            year_column = input("Please enter the correct column name for 'year': ").strip().lower()
        else:
            year_column = 'year'

        year = input("Enter the year to filter by: ").strip()
        if year_column in data.columns:
            return data[data[year_column] == int(year)]
        else:
            print(f"Error: Column '{year_column}' not found. Proceeding with full dataset.")
            return data

    elif subset_option == "random x":
        sample_size = int(input("Enter the number of random samples: ").strip())
        return data.sample(n=sample_size)

    elif subset_option == "no subset":
        return data

    else:
        print("Invalid option. Proceeding with full dataset.")
        return data


def main():
    # Load dataset
    default_file = r"C:\\Users\\spatt\\Desktop\\diss_3\\prodigy_custom\\data\\processed\\ungdc_chunk_model-v5_EntityContext.csv"
    file_path = input(f"Enter the path to the input file (or press Enter to use default): ").strip()
    if not file_path:
        file_path = default_file

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit()

    data = pd.read_csv(file_path)
    data = subset_data(data)

    if data.empty:
        print("No rows found for the specified subset. Exiting.")
        return

    results = []
    for index, row in data.iterrows():
        target = row['gpe_entity']
        text = row['text']

        print(f"\nProcessing row {index + 1}...")
        print(f"Original Text:\n{text}\n")  # Show original text
        prompt_outputs = analyze_with_prompts(co_client, target, text)

        # Collect adequacy ratings
        adequacy = {}
        for prompt_name, output in prompt_outputs.items():
            print(f"\nPrompt: {prompt_name}")
            print(f"Output: {output}")
            adequacy[prompt_name] = int(input(f"Is this output adequate? (1 for yes, 0 for no): ").strip())

        # Collect preference
        adequate_prompts = [k for k, v in adequacy.items() if v == 1]
        if adequate_prompts:
            print("\nAdequate prompts:")
            for prompt in adequate_prompts:
                print(f"- {prompt}")
            preferred = input("Which prompt do you prefer? (Enter prompt name or 'none'): ").strip()
        else:
            preferred = "none"

        # Save results
        row_result = {
            "Row": index + 1,
            **row.to_dict(),
            "Prompt Outputs": prompt_outputs,
            "Adequacy": adequacy,
            "Preferred Prompt": preferred
        }
        results.append(row_result)

        # Progress update
        print(f"\nCompleted row {index + 1}/{len(data)}")
        proceed = input("Do you want to proceed to the next row? (yes/no): ").strip().lower()
        if proceed != "yes":
            break

    # Save progress
    output_file = input("Enter the output file path and name (e.g., results.jsonl): ").strip()
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Evaluation complete. Results saved to {output_file}")


if __name__ == "__main__":
    main()
