import os
import cohere
import pandas as pd

def analyze_with_prompts(co_client, target, text):
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

        # Faithful Rewriting Prompts
        prompts = [
            {
                "name": "Original Faithful Rewriting",
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
                },
                "temperature": 0.7
            },
            {
                "name": "Lower Temperature Faithful Rewriting",
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
                },
                "temperature": 0.2
            },
            {
                "name": "Reworded Faithful Rewriting",
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
                },
                "temperature": 0.7
            },
            {
                "name": "Lower Temperature Reworded Faithful Rewriting",
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
                },
                "temperature": 0.2
            }
        ]

        results = {}

        # Run each prompt
        for prompt in prompts:
            print(f"Running {prompt['name']}...")
            response = co_client.chat(
                model="command-r7b-12-2024",
                messages=[system_message, prompt["message"]],
                temperature=prompt["temperature"]
            )
            results[prompt["name"]] = ''.join(segment.text for segment in response.message.content).strip()

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

    target_col = int(input("Select the column for 'Target' by number: ")) - 1
    text_col = int(input("Select the column for 'Text' by number: ")) - 1

    # Iterate over dataset
    results = []
    for index, row in data.iterrows():
        target = row[columns[target_col]]
        text = row[columns[text_col]]

        print(f"\nProcessing row {index + 1}...")
        result = analyze_with_prompts(co_client, target, text)
        results.append(result)

        print(f"\nResults for row {index + 1}:")
        for key, value in result.items():
            print(f"{key}: {value}")

        if input("Would you like to proceed to the next row? (yes/no): ").strip().lower() != "yes":
            break

    # Output results
    output_file = "analysis_results_v3.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Analysis complete. Results saved to {output_file}")
