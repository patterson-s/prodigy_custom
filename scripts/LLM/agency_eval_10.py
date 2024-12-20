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
            "name": "Preamble + Instruction variance",
            "temperature": 0.7,
            "message": {
                "role": "user",
                "content": (
                    f"Summarize how the target state is discussed in the following text in a single sentence.\n\n"
                    f"Target: {target}\n"
                    f"Text: {text}\n\n"
                    f"Instructions:\n"
                    f"1. Begin the sentence by identifying the target state.\n"
                    f"2. Write the summary in the active voice.\n"
                    f"3. Ensure that the meaning of the summary reflects the original context.\n"
                    f"4. Be sure to accurately portray who is doing what to whom."
                )
            }
        },
        {
            "name": "mod1",
            "temperature": 0.7,
            "message": {
                "role": "user",
                "content": (
                    f"Summarize how the target state is discussed in the following text in a single sentence.\n\n"
                    f"Target: {target}\n"
                    f"Text: {text}\n\n"
                    f"Instructions:\n"
                    f"1. Begin with '{target} is mentioned' or '{target} is discussed'\n"
                    f"2. Continue the sentence by describing how they are mentioned/discussed\n"
                    f"3. Ensure that the meaning of the summary reflects the original context.\n"
                    f"4. Be sure to accurately portray who is doing what to whom."
                )
            }
        },
        {
            "name": "Preamble + Instruction variance + Criteria",
            "temperature": 0.7,
            "message": {
                "role": "user",
                "content": (
                    f"Summarize how the target state is discussed in the following text in a single sentence.\n\n"
                    f"Target: {target}\n"
                    f"Text: {text}\n\n"
                    f"Instructions:\n"
                    f"1. Begin the sentence by identifying the target state.\n"
                    f"2. Write the summary in the active voice.\n"
                    f"3. Ensure that the meaning of the summary reflects the original context.\n"
                    f"4. Be sure to accurately portray who is doing what to whom.\n\n"
                    f"You will be evaluated based on the following criteria:\n"
                    f"- The output must explicitly align with the geopolitical or thematic context, accurately reflecting the target state’s role and relationships.\n"
                    f"- The response should provide a nuanced summary that captures essential details without omitting critical context.\n"
                    f"- Language must be unambiguous, with precise terms and a direct explanation of how the target state is discussed.\n"
                    f"- The output must demonstrate polished organization, a professional tone, and consistent adherence to the prompt's structure.\n"
                    f"- Ideas must flow logically, clearly connecting actions, entities, and their relationships without introducing contradictions."
                )
            }
        },
        {
            "name": "mod1 + Criteria",
            "temperature": 0.7,
            "message": {
                "role": "user",
                "content": (
                    f"Summarize how the target state is discussed in the following text in a single sentence.\n\n"
                    f"Target: {target}\n"
                    f"Text: {text}\n\n"
                    f"Instructions:\n"
                    f"1. Begin with '{target} is mentioned' or '{target} is discussed'\n"
                    f"2. Continue the sentence by describing how they are mentioned/discussed\n"
                    f"3. Ensure that the meaning of the summary reflects the original context.\n"
                    f"4. Be sure to accurately portray who is doing what to whom.\n\n"
                    f"You will be evaluated based on the following criteria:\n"
                    f"- The output must explicitly align with the geopolitical or thematic context, accurately reflecting the target state’s role and relationships.\n"
                    f"- The response should provide a nuanced summary that captures essential details without omitting critical context.\n"
                    f"- Language must be unambiguous, with precise terms and a direct explanation of how the target state is discussed.\n"
                    f"- The output must demonstrate polished organization, a professional tone, and consistent adherence to the prompt's structure.\n"
                    f"- Ideas must flow logically, clearly connecting actions, entities, and their relationships without introducing contradictions."
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

def load_previous_results(output_folder):
    """Load previous intermediate results if they exist."""
    results = []
    output_file = os.path.join(output_folder, "intermediate_results.jsonl")
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                results.append(json.loads(line.strip()))
        print(f"Resuming from previously saved results in {output_file}")
    return results

def main():
    # Initialize Cohere client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    co_client = cohere.ClientV2(api_key)

    # Get output folder at the start
    output_folder = input("Enter the output folder path: ").strip()
    os.makedirs(output_folder, exist_ok=True)

    # Load previous results if available
    results = load_previous_results(output_folder)
    completed_rows = {result["Row"] for result in results}

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

    for index, row in data.iterrows():
        if index + 1 in completed_rows:
            continue  # Skip rows that have already been processed

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
            while True:
                try:
                    adequacy[prompt_name] = int(input(f"Is this output adequate? (1 for yes, 0 for no): ").strip())
                    break
                except ValueError:
                    print("Invalid input. Please enter 1 or 0.")

        # Collect preference
        adequate_prompts = [k for k, v in adequacy.items() if v == 1]
        if adequate_prompts:
            print("\nOptions:")
            for i, prompt in enumerate(adequate_prompts, start=1):
                print(f"{i}: {prompt}")
            # Add the new options after the prompt options
            next_num = len(adequate_prompts) + 1
            print(f"{next_num}: All prompts equally good")
            print(f"{next_num + 1}: All prompts equally bad")
            print("0: None preferred")
            
            while True:
                preferred_input = input("Which option do you prefer? Enter number: ").strip()
                try:
                    choice_num = int(preferred_input)
                    if choice_num == 0:
                        preferred = "none"
                        break
                    elif 1 <= choice_num <= len(adequate_prompts):
                        preferred = adequate_prompts[choice_num - 1]
                        break
                    elif choice_num == next_num:
                        preferred = "all_equal_good"
                        break
                    elif choice_num == next_num + 1:
                        preferred = "all_equal_bad"
                        break
                    else:
                        print(f"Please enter a number between 0 and {next_num + 1}")
                except ValueError:
                    print("Invalid input. Please enter a number")
        else:
            preferred = "none"

        # Add notes option
        print("\nadd note:")
        note = input().strip()

        # Save results
        row_result = {
            "Row": index + 1,
            **row.to_dict(),
            "Prompt Outputs": prompt_outputs,
            "Adequacy": adequacy,
            "Preferred Prompt": preferred,
            "Note": note  # Add the note to the results
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