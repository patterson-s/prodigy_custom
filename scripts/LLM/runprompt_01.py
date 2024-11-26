import os
import json
import cohere


def list_files(directory, extension):
    """
    List all files with a specific extension in the given directory.
    """
    return [f for f in os.listdir(directory) if f.endswith(extension)]


def choose_file(files, file_type):
    """
    Display the available files and allow the user to select one.
    """
    print(f"Available {file_type.capitalize()} Files:")
    for idx, file in enumerate(files, start=1):
        print(f"{idx}: {file}")
    while True:
        try:
            choice = int(input(f"Enter the number of the {file_type} you want to use: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print(f"Invalid choice. Please enter a number between 1 and {len(files)}.")
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

    # Directories containing prompts and metadata
    prompts_dir = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\prompts"
    metadata_dir = prompts_dir  # Assuming metadata is in the same folder as prompts

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

    # Validate metadata fields
    required_keys = [
        "model", "max_tokens", "temperature", "k", "p",
        "frequency_penalty", "presence_penalty", "stop_sequences", "return_likelihoods"
    ]
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Missing required metadata field: {key}")

    # Run the prompt with the selected metadata
    result = run_prompt(co_client, prompt, metadata)
    print("\nPrompt Result:")
    print(result)


if __name__ == "__main__":
    main()
