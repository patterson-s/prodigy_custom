import os
import cohere
import pandas as pd

def analyze_with_prompts(co_client, source, target, year, text):
    try:
        print("\nPreparing analysis...")

        # Common system message for all prompts
        system_message = {
            "role": "system",
            "content": (
                "You are an advanced language model trained to analyze diplomatic speech. "
                "Your task is to process the provided text based on the user's instructions."
            )
        }

        # Prompt 1: Contextual Summarization (Political Analysis)
        user_message1 = {
            "role": "user",
            "content": (
                f"Summarize how the target state is discussed in the provided text.\n\n"
                f"Target: {target}\n"
                f"Source: {source}\n"
                f"Year: {year}\n"
                f"Text: {text}\n\n"
                f"Instructions:\n"
                f"1. Start your output sentence with the target state.\n"
                f"2. Use precise and concise language to describe the context, actions, or relationships involving the target state.\n"
                f"3. Focus only on the information directly relevant to the target state, omitting unrelated details."
            )
        }

        # Prompt 2: Linguistic Contextualization (Faithful Rewriting)
        user_message2 = {
            "role": "user",
            "content": (
                f"Rewrite the context in which the target state is mentioned into a single sentence.\n\n"
                f"Target: {target}\n"
                f"Source: {source}\n"
                f"Year: {year}\n"
                f"Text: {text}\n\n"
                f"Instructions:\n"
                f"1. Begin with the target state.\n"
                f"2. Use as much of the original language as possible.\n"
                f"3. Ensure grammatical correctness by addressing issues such as conjugations or sentence structure.\n"
                f"4. Preserve the essence and meaning of the original text."
            )
        }

        # Function to run a specific prompt with a specific model
        def run_prompt(prompt, model):
            response = co_client.chat(
                model=model,
                messages=[system_message, prompt]
            )
            # Extract text from each content item and join them
            return ''.join([segment.text for segment in response.message.content]).strip()

        results = {}

        print("Running Prompt 1 (Political Analysis) with r7b...")
        results["Prompt 1 - r7b"] = run_prompt(user_message1, "command-r7b-12-2024")
        print(results["Prompt 1 - r7b"])

        print("Running Prompt 1 (Political Analysis) with r-plus...")
        results["Prompt 1 - r-plus"] = run_prompt(user_message1, "command-r-plus")
        print(results["Prompt 1 - r-plus"])

        print("Running Prompt 2 (Faithful Rewriting) with r7b...")
        results["Prompt 2 - r7b"] = run_prompt(user_message2, "command-r7b-12-2024")
        print(results["Prompt 2 - r7b"])

        print("Running Prompt 2 (Faithful Rewriting) with r-plus...")
        results["Prompt 2 - r-plus"] = run_prompt(user_message2, "command-r-plus")
        print(results["Prompt 2 - r-plus"])

        return results

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        return {"Error": str(e)}

if __name__ == "__main__":
    # Initialize Cohere client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    co_client = cohere.ClientV2(api_key)

    # Input file selection
    default_file = r"C:\\Users\\spatt\\Desktop\\diss_3\\prodigy_custom\\data\\processed\\ungdc_chunk_model-v5_EntityContext.csv"
    file_path = input(f"Enter the path to the input file (or press Enter to use default): ").strip()
    if not file_path:
        file_path = default_file

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit()

    # Load dataset
    data = pd.read_csv(file_path)
    columns = list(data.columns)

    # Display columns and get variable selections
    print("\nAvailable columns:")
    for i, col in enumerate(columns):
        print(f"{i + 1}: {col}")

    source_col = int(input("Select the column for 'Source' by number: ")) - 1
    target_col = int(input("Select the column for 'Target' by number: ")) - 1
    year_col = int(input("Select the column for 'Year' by number: ")) - 1
    text_col = int(input("Select the column for 'Text' by number: ")) - 1

    # Iterate over dataset
    results = []
    for index, row in data.iterrows():
        source = row[columns[source_col]]
        target = row[columns[target_col]]
        year = row[columns[year_col]]
        text = row[columns[text_col]]

        print(f"\nProcessing row {index + 1}...")
        result = analyze_with_prompts(co_client, source, target, year, text)
        results.append(result)

        print("\nResults for row {index + 1}:")
        for key, value in result.items():
            print(f"{key}: {value}")

        if input("Would you like to proceed to the next row? (yes/no): ").strip().lower() != "yes":
            break

    # Output results
    output_file = "analysis_results.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")
