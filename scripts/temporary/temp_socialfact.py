import os
import json

# Define paths
input_path_1946 = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\socialfact_batch_03"
input_path_1947 = os.path.join(input_path_1946, "1947")
output_path = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\analysis"
output_file_combined = os.path.join(output_path, "socialfact_batch_03_1946-7.jsonl")
output_file_filtered = os.path.join(output_path, "socialfact_batch_03_1946-7_filtered.jsonl")

# Fields to include in the filtered version
filtered_fields = ["doc_id", "source", "target", "year", "row", "text", "extracted_output"]

def load_jsonl_files(folder):
    """Load and combine JSONL files from a folder."""
    data = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jsonl"):
            filepath = os.path.join(folder, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                for line in file:
                    data.append(json.loads(line))
    return data

def filter_data(data, fields):
    """Filter data to include only specific fields."""
    filtered = []
    for entry in data:
        filtered_entry = {field: entry.get(field, None) for field in fields}
        filtered.append(filtered_entry)
    return filtered

def save_jsonl(data, filepath):
    """Save data to a JSONL file."""
    with open(filepath, "w", encoding="utf-8") as file:
        for entry in data:
            file.write(json.dumps(entry) + "\n")

def main():
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Load and combine data
    data_1946 = load_jsonl_files(input_path_1946)
    data_1947 = load_jsonl_files(input_path_1947)
    combined_data = data_1946 + data_1947
    
    # Save combined dataset
    save_jsonl(combined_data, output_file_combined)
    print(f"Combined file saved to {output_file_combined}")
    
    # Filter data and save
    filtered_data = filter_data(combined_data, filtered_fields)
    save_jsonl(filtered_data, output_file_filtered)
    print(f"Filtered file saved to {output_file_filtered}")

if __name__ == "__main__":
    main()
