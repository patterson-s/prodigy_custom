#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess
import spacy
import json
import os

def load_model_with_patterns(model_path, patterns_path, output_model_path):
    """
    Load the specified spaCy model, add an EntityRuler with patterns, and save the modified model.
    """
    # Load the spaCy model
    nlp = spacy.load(model_path)
    
    # Add the EntityRuler before the NER, configured to overwrite entities based on patterns
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
    else:
        ruler = nlp.get_pipe("entity_ruler")

    # Load and add patterns from the JSONL patterns file
    with open(patterns_path, "r", encoding="utf-8") as f:
        patterns = [json.loads(line) for line in f]
    ruler.add_patterns(patterns)

    # Save the modified model with EntityRuler to the specified output path
    nlp.to_disk(output_model_path)
    print(f"Model with EntityRuler saved to {output_model_path}")

def main():
    # Get user input for entity type and dataset name
    entity_type = input("Enter the entity type (e.g., institution): ").upper()
    dataset_name = input("Enter the Prodigy dataset name: ")

    # Define paths
    base_model_path = "en_core_web_lg"  # Base model path
    patterns_path = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/patterns/institution_pattern_01.jsonl"
    output_model_path = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/models/en_core_web_lg_with_ruler_manual"
    sample_file_path = f"C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/annotation_samples/{entity_type.lower()}_samples_01.jsonl"

    # Ensure patterns file exists
    if not os.path.exists(patterns_path):
        print(f"Patterns file not found: {patterns_path}")
        return

    # Load model, add EntityRuler, and save the updated model
    load_model_with_patterns(base_model_path, patterns_path, output_model_path)

    # Prodigy command to start ner.manual
    command = [
        "python", "-m", "prodigy", "ner.manual", dataset_name,
        output_model_path, sample_file_path, "--label", entity_type
    ]

    print("Starting Prodigy with the following command:")
    print(" ".join(command))

    # Run the Prodigy command
    subprocess.run(command)

if __name__ == "__main__":
    main()
