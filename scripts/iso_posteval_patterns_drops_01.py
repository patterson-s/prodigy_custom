import pandas as pd
import json
import os

def process_csv(input_csv, output_pattern_jsonl, output_drop_jsonl):
    """
    Process a CSV file to deduplicate entries and split into two JSONL files.
    One file contains patterns to keep, and the other contains drops.
    
    Args:
        input_csv (str): Path to the input CSV file.
        output_pattern_jsonl (str): Path to save the patterns JSONL.
        output_drop_jsonl (str): Path to save the drops JSONL.
    """
    # Load the CSV file
    print(f"Loading file: {input_csv}")
    data = pd.read_csv(input_csv)

    # Deduplicate the data
    print("Deduplicating entries...")
    data = data.drop_duplicates()

    # Split the data based on "drop" in the 'state' or 'ISO_Code' column
    print("Splitting data into patterns and drops...")
    drop_data = data[(data['state'] == 'drop') | (data['ISO_Code'] == 'drop')]
    pattern_data = data[~((data['state'] == 'drop') | (data['ISO_Code'] == 'drop'))]

    # Save patterns to a JSONL file
    with open(output_pattern_jsonl, 'w') as f_pattern:
        for _, row in pattern_data.iterrows():
            json.dump(row.to_dict(), f_pattern)
            f_pattern.write('\n')

    # Save drops to a JSONL file
    with open(output_drop_jsonl, 'w') as f_drop:
        for _, row in drop_data.iterrows():
            json.dump(row.to_dict(), f_drop)
            f_drop.write('\n')

    print(f"Files saved: {output_pattern_jsonl} and {output_drop_jsonl}")

def merge_jsonl(new_file, master_file):
    """
    Merge a new JSONL file into the master JSONL file, deduplicating entries.
    
    Args:
        new_file (str): Path to the new JSONL file.
        master_file (str): Path to the master JSONL file.
    """
    print(f"Merging {new_file} into {master_file}...")
    
    # Load the existing master file if it exists
    if os.path.exists(master_file):
        with open(master_file, 'r') as f_master:
            master_data = [json.loads(line) for line in f_master]
    else:
        master_data = []

    # Load the new data
    with open(new_file, 'r') as f_new:
        new_data = [json.loads(line) for line in f_new]

    # Combine and deduplicate
    combined_data = pd.DataFrame(master_data + new_data).drop_duplicates().to_dict(orient='records')

    # Write back to the master file
    with open(master_file, 'w') as f_master:
        for entry in combined_data:
            json.dump(entry, f_master)
            f_master.write('\n')

    print(f"Merge complete. {len(combined_data)} total entries in {master_file}.")

def main():
    base_path = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\evaluation"
    output_dir = os.path.join(base_path, "patterns_drops")
    patterns_master_file = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\patterns\iso_match_master_01.jsonl"
    drops_master_file = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\patterns\iso_drop_master_01.jsonl"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Enter the name of the CSV file (without the .csv extension):")
    file_name = input().strip()

    # Construct full file path
    input_csv_path = os.path.join(base_path, f"{file_name}.csv")
    if not os.path.exists(input_csv_path):
        print(f"Error: File '{input_csv_path}' does not exist.")
        return

    # Define output JSONL paths
    output_pattern_path = os.path.join(output_dir, f"{file_name}_patterns.jsonl")
    output_drop_path = os.path.join(output_dir, f"{file_name}_drops.jsonl")

    # Process the file
    process_csv(input_csv_path, output_pattern_path, output_drop_path)

    # Ask if user wants to merge with master files
    print("\nWould you like to merge these files into the master patterns and drops files? (y/n):")
    choice = input().strip().lower()
    if choice == 'y':
        merge_jsonl(output_pattern_path, patterns_master_file)
        merge_jsonl(output_drop_path, drops_master_file)
    else:
        print("Merge skipped. You can manually review the files before merging.")

if __name__ == "__main__":
    main()
