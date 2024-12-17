import os
import cohere
import pandas as pd
import json

def analyze_with_prompts(co_client, target, text):
    """Run prompts with temperature variations and return their outputs."""
    system_message = {
        "role": "system",
        "content": (
            "You are an advanced language model trained to analyze diplomatic speech. "
            "Your task is to process the provided text based on the user's instructions."
        )
    }

    # Define prompts with temperature variations
    prompts = [
        {
            "name": "Prompt 1.1 (temp=0.7)",
            "temperature": 0.7,
            "message": {
                "role": "user",
                "content": (
                    f"Rewrite the context in which the target state is mentioned into a single sentence.\n\n"
                    f"Target: {target}\n"
                    f"Text: {text}\n\n"
                    f"Instructions:\n"
                    f"1. Begin with the target state.\n"
                    f"2. Use as much of the original language as possible.\n"
                    f"3. Ensure grammatical correctness by addressing issues such as conjugations or sentence structure.\n"
                    f"4. Preserve the essence and meaning of the original text."
                )
            }
        },
        {
            "name": "Prompt 1.2 (temp=0.2)",
            "temperature": 0.2,
            "message": {
                "role": "user",
                "content": (
                    f"Rewrite the context in which the target state is mentioned into a single sentence.\n\n"
                    f"Target: {target}\n"
                    f"Text: {text}\n\n"
                    f"Instructions:\n"
                    f"1. Begin with the target state.\n"
                    f"2. Use as much of the original language as possible.\n"
                    f"3. Ensure grammatical correctness by addressing issues such as conjugations or sentence structure.\n"
                    f"4. Preserve the essence and meaning of the original text."
                )
            }
        },
        {
            "name": "Prompt 2.1 (temp=0.7)",
            "temperature": 0.7,
            "message": {
                "role": "user",
                "content": (
                    f"Summarize the mention of the target state in the text using original language.\n\n"
                    f"Target: {target}\n"
                    f"Text: {text}\n\n"
                    f"Instructions:\n"
                    f"1. Start with the target state.\n"
                    f"2. Use concise, original wording where possible.\n"
                    f"3. Maintain grammatical accuracy.\n"
                    f"4. Ensure the meaning reflects the original context."
                )
            }
        },
        {
            "name": "Prompt 2.2 (temp=0.2)",
            "temperature": 0.2,
            "message": {
                "role": "user",
                "content": (
                    f"Summarize the mention of the target state in the text using original language.\n\n"
                    f"Target: {target}\n"
                    f"Text: {text}\n\n"
                    f"Instructions:\n"
                    f"1. Start with the target state.\n"
                    f"2. Use concise, original wording where possible.\n"
                    f"3. Maintain grammatical accuracy.\n"
                    f"4. Ensure the meaning reflects the original context."
                )
            }
        }
    ]

    results = {}
    for prompt in prompts:
        try:
            print(f"Running {prompt['name']}...")
            response = co_client.chat(
                model="command-r7b-12-2024",
                messages=[system_message, prompt["message"]],
                temperature=prompt["temperature"]
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

def save_intermediate_results(results, output_folder):
    """Save intermediate results to the output folder."""
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, "intermediate_results.jsonl")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Intermediate results saved to {output_file}")

def main():
    # Initialize Cohere client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    co_client = cohere.ClientV2(api_key)

    # Get output folder at the start
    output_folder = input("Enter the output folder path: ").strip()
    os.makedirs(output_folder, exist_ok=True)

    # Load dataset
    default_file = r"C:\\Users\\spatt\\Desktop\\diss_3\\prodigy_custom\\data\\processed\\ungdc_chunk_model-v5_EntityContext.csv"
    file_path = input(f"Enter the path to the input file (or press Enter to use default): ").strip()
    if not file_path:
        file_path = default_file

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit()

    data = pd.read_csv(file_path)

    # Apply subsetting logic
    data = subset_data(data)

    # Ask user if self-mentions should be included
    include_self_mentions = input("Do you want to include self-mentions? (yes/no): ").strip().lower()
    if include_self_mentions == "no":
        if 'iso' in data.columns and 'iso_code' in data.columns:
            data = data[data['iso'] != data['iso_code']]
        else:
            print("Columns 'iso' and 'ISO_Code' not found. Proceeding with full dataset.")

    results = []
    for index, row in data.iterrows():
        target = row['gpe_entity']
        text = row['text']

        print(f"\nProcessing row {index + 1}...")
        print(f"Original Text:\n{text}\n")  # Show original text
        prompt_outputs = analyze_with_prompts(co_client, target, text)

        # Collect adequacy ratings
        adequacy = {}
        for i, (prompt_name, output) in enumerate(prompt_outputs.items(), start=1):
            print(f"\nPrompt {i}: {prompt_name}")
            print(f"Output: {output}")
            adequacy[prompt_name] = int(input(f"Is this output adequate? (1 for yes, 0 for no): ").strip())

        # Collect preference
        adequate_prompts = [k for k, v in adequacy.items() if v == 1]
        if adequate_prompts:
            print("\nAdequate prompts:")
            for i, prompt in enumerate(adequate_prompts, start=1):
                print(f"{i}: {prompt}")
            preferred_index = input("Which prompt do you prefer? (Enter number or '0' for none): ").strip()
            preferred = adequate_prompts[int(preferred_index) - 1] if preferred_index != '0' else "none"
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

        # Save intermediate results
        save_intermediate_results(results, output_folder)

        # Progress update
        print(f"\nCompleted row {index + 1}/{len(data)}")
        proceed = input("Do you want to proceed to the next row? (yes/no): ").strip().lower()
        if proceed != "yes":
            break

    # Final save
    final_output_file = os.path.join(output_folder, "final_results.jsonl")
    with open(final_output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"Evaluation complete. Final results saved to {final_output_file}")

if __name__ == "__main__":
    main()
