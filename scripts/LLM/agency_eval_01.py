import os
import cohere

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
            # Extract and concatenate the content from response
            return ''.join([segment.text for segment in response.message.content]).strip()

        # Running all four prompts
        print("Running Prompt 1 (Political Analysis) with r7b...")
        result1_r7b = run_prompt(user_message1, "command-r7b-12-2024")

        print("Running Prompt 1 (Political Analysis) with r-plus...")
        result1_r_plus = run_prompt(user_message1, "command-r-plus")

        print("Running Prompt 2 (Faithful Rewriting) with r7b...")
        result2_r7b = run_prompt(user_message2, "command-r7b-12-2024")

        print("Running Prompt 2 (Faithful Rewriting) with r-plus...")
        result2_r_plus = run_prompt(user_message2, "command-r-plus")

        # Display results
        print("\nAnalysis Results:")
        print("-" * 50)
        print("Prompt 1 (Political Analysis) - r7b:")
        print(result1_r7b)
        print("-" * 50)
        print("Prompt 1 (Political Analysis) - r-plus:")
        print(result1_r_plus)
        print("-" * 50)
        print("Prompt 2 (Faithful Rewriting) - r7b:")
        print(result2_r7b)
        print("-" * 50)
        print("Prompt 2 (Faithful Rewriting) - r-plus:")
        print(result2_r_plus)
        print("-" * 50 + "\n")

        return {
            "Prompt 1 - r7b": result1_r7b,
            "Prompt 1 - r-plus": result1_r_plus,
            "Prompt 2 - r7b": result2_r7b,
            "Prompt 2 - r-plus": result2_r_plus
        }

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        return {"Error": str(e)}

if __name__ == "__main__":
    # Initialize Cohere client with new V2 client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    co_client = cohere.ClientV2(api_key)

    print("Welcome to the Diplomatic Speech Four-Prompt Analysis Demo!")
    print("(Type 'exit' at any time to quit)\n")

    while True:
        try:
            source = input("Enter source country: ").strip()
            if source.lower() == "exit":
                print("Goodbye!")
                break

            target = input("Enter target country: ").strip()
            if target.lower() == "exit":
                print("Goodbye!")
                break

            year = input("Enter year: ").strip()
            if year.lower() == "exit":
                print("Goodbye!")
                break

            print("\nEnter speech text (type END on a new line when finished):")
            text_lines = []
            while True:
                line = input().strip()
                if line.upper() == 'END':
                    break
                if line.lower() == 'exit':
                    print("Goodbye!")
                    exit()
                text_lines.append(line)

            if not text_lines:
                print("No text entered. Please try again.")
                continue

            text = " ".join(text_lines)
            print("\nProcessing your input...")
            results = analyze_with_prompts(co_client, source, target, year, text)

            retry = input("\nWould you like to try another analysis? (yes/no): ").strip().lower()
            if retry != "yes":
                print("Goodbye!")
                break

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            retry = input("\nWould you like to try again? (yes/no): ").strip().lower()
            if retry != "yes":
                print("Goodbye!")
                break
