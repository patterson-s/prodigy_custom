#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
start_annotation_01.py

This script starts a Prodigy ner.correct session using a model with an
EntityRuler that incorporates patterns for specific entity types.
"""

import os
import sys
import json
import spacy
from spacy.pipeline import EntityRuler
from pathlib import Path
import subprocess

# Set file paths directly in the script
MODEL_PATH = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/models/en_core_web_lg_with_ruler"
PATTERNS_PATH = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/patterns/institution_pattern_01.jsonl"
OUTPUT_DIR = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/annotation_samples"

def load_model_with_patterns(model_path, patterns_path):
    # Load spaCy model
    nlp = spacy.load("en_core_web_lg")
    
    # Add an EntityRuler with patterns before the NER component
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")

    # Load patterns from the updated JSONL patterns file
    with open(patterns_path, "r", encoding="utf-8") as f:
        patterns = [json.loads(line) for line in f]
    ruler.add_patterns(patterns)
    
    # Save model with EntityRuler applied
    nlp.to_disk(model_path)
    return model_path

    
    # Save model with EntityRuler
    nlp.to_disk(model_path)
    return model_path

def get_latest_sample_file(entity_type):
    # Get the list of sample files for the entity type
    sample_files = list(Path(OUTPUT_DIR).glob(f"{entity_type}_samples_*.jsonl"))
    if not sample_files:
        print(f"No sample files found for entity type '{entity_type}' in {OUTPUT_DIR}.")
        sys.exit(1)
    # Sort the files to get the latest one
    latest_file = max(sample_files, key=os.path.getctime)
    return latest_file

def main():
    # Prompt the user for entity_type and dataset name
    entity_type = input("Enter the entity type (e.g., institution): ").strip()
    dataset_name = input("Enter the Prodigy dataset name: ").strip()

    # Load model with patterns and save it
    model_with_ruler = load_model_with_patterns(MODEL_PATH, PATTERNS_PATH)
    
    # Get the latest sample file for the entity type
    sample_file = get_latest_sample_file(entity_type)

    # Construct the Prodigy command for ner.correct
    command = [
        "python", "-m", "prodigy",
        "ner.correct",
        dataset_name,
        model_with_ruler,
        str(sample_file),
        "--label",
        entity_type.upper()
    ]

    # Print the command for debugging purposes
    print("Starting Prodigy with the following command:")
    print(" ".join(command))

    # Start the Prodigy ner.correct session
    subprocess.run(command)

if __name__ == "__main__":
    main()
