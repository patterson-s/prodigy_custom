import os
import json
import pandas as pd
import cohere


def list_files(directory, extension):
    """
    List all files with a specific extension in the given directory.
    """
    return [f for f in os.listdir(directory) if f.endswith(extension)]


def choose_file(files, file_type, allow_skip=False):
    """
    Display the available files and allow the user to select one.
    """
    print(f"Available {file_type.capitalize()} Files:")
    if allow_skip:
        print("0: Skip (No dataset)")

    for idx, file in enumerate(files, start=1):
        print(f"{idx}: {file}")

    while True:
        try:
            choice = int(input(f"Enter the number of the {file_type} you want to use: "))
            if allow_skip and choice == 0:
                return None
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print(f"Invalid choice. Please enter a number between 0 and {len(files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def load_file(filepath):
    """
    Load content from a text file.
    """
    with open(filepath, 'r') as file:
        return file.read()


def load_metadata(filepath):
    """
    Load metadata from a JSON file.
    """
    with open(filepath, 'r') as file:
        return json.load(file)


def load_dataset(filepath):
    """
    Load a dataset from a CSV file into a pandas DataFrame.
    """
    return pd.read_csv(filepath)


def run_prompt(co_client, prompt, metadata):
    """
    Run the prompt with Cohere API using parameters from metadata.
    """
    if not metadata:
        raise ValueError("Metadata is required to execute the prompt.")

    # Pass parameters directly from metadata
    response = co_client.generate(
        model=metadata["model"],
        prompt=prompt,
        max_tokens=metadata["max_tokens"],
        temperature=metadata["temperature"],
        k=metadata["k"],
        p=metadata["p"],
        frequency_penalty=metadata["frequency_penalty"],
        presence_penalty=metadata["presence_penalty"],
        stop_sequences=metadata["stop_sequences"],
        return_likelihoods=metadata["return_likelihoods"]
    )
    return response.generations[0].text.strip()


def main():
    # Initialize the Cohere client
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("Cohere API key not found. Please ensure it's set in the environment.")
    co_client = cohere.Client(api_key)

    # Directories for prompts, metadata, and datasets
    prompts_dir = r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/prompts"
    metadata_dir = prompts_dir  # Assuming metadata is in the same folder as prompts
    datasets_dir = r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed"

    # List and choose the prompt file
    prompt_files = list_files(prompts_dir, ".txt")
    if not prompt_files:
        print("No prompt files found.")
        return
    selected_prompt = choose_file(prompt_files, "prompt")
    prompt_path = os.path.join(prompts_dir, selected_prompt)
    prompt = load_file(prompt_path)

    # List and choose the metadata file
    metadata_files = list_files(metadata_dir, ".json")
    if not metadata_files:
        print("No metadata files found.")
        return
    selected_metadata = choose_file(metadata_files, "metadata")
    metadata_path = os.path.join(metadata_dir, selected_metadata)
    metadata = load_metadata(metadata_path)

    # List and choose the dataset file
    dataset_files = list_files(datasets_dir, ".csv")
    selected_dataset = choose_file(dataset_files, "dataset", allow_skip=True)
    dataset = None
    if selected_dataset:
        dataset_path = os.path.join(datasets_dir, selected_dataset)
        dataset = load_dataset(dataset_path)

    # If no dataset, just run the prompt once
    if dataset is None:
        result = run_prompt(co_client, prompt, metadata)
        print("\nPrompt Result:")
        print(result)
    else:
        # Run the prompt on each row of the dataset
        print(f"\nRunning prompt on dataset: {selected_dataset}")
        column_name = input("Enter the column name to use for the prompt input: ")
        if column_name not in dataset.columns:
            print(f"Column '{column_name}' not found in the dataset.")
            return

        results = []
        for idx, row in dataset.iterrows():
            input_text = row[column_name]
            prompt_instance = prompt.format(input_text=input_text)
            print(f"\nProcessing row {idx + 1}...")
            result = run_prompt(co_client, prompt_instance, metadata)
            results.append({
                "row_index": idx,
                "input_text": input_text,
                "output": result
            })
            print(f"Result for row {idx + 1}: {result}")

        # Save results to a new CSV file
        results_df = pd.DataFrame(results)
        output_file = os.path.join(datasets_dir, f"results_{selected_dataset}")
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
