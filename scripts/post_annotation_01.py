import subprocess
import os
import json

def get_next_annotation_filename(entity_type, annotations_dir):
    """
    Generates the next available annotation filename based on existing files in the annotations directory.
    """
    prefix = f"{entity_type}_annotation_"
    existing_files = [f for f in os.listdir(annotations_dir) if f.startswith(prefix) and f.endswith(".jsonl")]
    
    # Find the highest version number and increment
    if existing_files:
        existing_versions = [int(f.split("_")[-1].split(".")[0]) for f in existing_files]
        next_version = max(existing_versions) + 1
    else:
        next_version = 1
    
    return os.path.join(annotations_dir, f"{entity_type}_annotation_{next_version:02}.jsonl")

def export_annotations(dataset_name, entity_type, annotations_dir):
    """
    Exports Prodigy annotations for a specified dataset and saves them with a systematic filename.
    """
    # Ensure the annotations directory exists
    os.makedirs(annotations_dir, exist_ok=True)
    
    # Determine the filename for the next annotation file
    output_file = get_next_annotation_filename(entity_type, annotations_dir)
    
    # Run the Prodigy db-out command
    print(f"Exporting annotations for dataset '{dataset_name}' to '{output_file}'...")
    with open(output_file, 'w', encoding='utf-8') as f:
        subprocess.run(["python", "-m", "prodigy", "db-out", dataset_name], stdout=f)
    
    print(f"Annotations exported successfully to {output_file}")

def main():
    # Prompt user for entity type and dataset name
    entity_type = input("Enter the entity type (e.g., institution): ").lower()
    dataset_name = input("Enter the Prodigy dataset name: ")
    
    # Define the path to the annotations directory
    annotations_dir = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/annotations"
    
    # Export the annotations
    export_annotations(dataset_name, entity_type, annotations_dir)

if __name__ == "__main__":
    main()
