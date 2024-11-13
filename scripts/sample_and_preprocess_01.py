#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sample_and_preprocess_01.py

This script samples a specified number of texts from a raw dataset,
preprocesses them, and saves them for annotation with an incremented
filename based on the entity type. It also logs sampled items to prevent
duplicates in future sampling rounds and uses a sentencizer for sentence
segmentation.
"""

import random
import os
import json
import csv
import spacy
from pathlib import Path

# Set file paths directly in the script
SOURCE_FILE = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/raw/ungdc_1946-2022.csv"
OUTPUT_DIR = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/annotation_samples"
LOG_DIR = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/logs"

# Load spaCy model with sentencizer
nlp = spacy.blank("en")
if "sentencizer" not in nlp.pipe_names:
    nlp.add_pipe("sentencizer")

def load_existing_log(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logged_ids = set(json.load(f))
        return logged_ids
    else:
        return set()

def update_log(log_file, new_ids):
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            logged_ids = set(json.load(f))
    else:
        logged_ids = set()
    logged_ids.update(new_ids)
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(list(logged_ids), f)

def sample_data(num_samples, log_file):
    # Load the raw data
    data = []
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)

    # Load existing log to avoid duplicates
    logged_ids = load_existing_log(log_file)
    available_data = [d for d in data if str(d.get('doc_id')) not in logged_ids]

    if len(available_data) < num_samples:
        raise ValueError("Not enough unsampled data available.")

    # Sample without replacement
    sampled_data = random.sample(available_data, num_samples)
    sampled_ids = [str(d.get('doc_id')) for d in sampled_data]

    # Update the log with new sampled IDs
    update_log(log_file, sampled_ids)

    return sampled_data

def preprocess_text(text):
    """Clean text by removing unnecessary line breaks and segmenting into sentences."""
    # Replace single line breaks with spaces
    cleaned_text = text.replace("\n", " ")
    
    # Run through the model's sentencizer for sentence segmentation
    doc = nlp(cleaned_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    
    # Fall back on basic punctuation-based segmentation if needed
    if len(sentences) <= 1:  # If sentencizer didn't split well
        sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', cleaned_text) if s]
    
    return sentences


def get_next_filename(entity_type, output_dir):
    # Check for existing files and increment the filename version
    existing_files = list(Path(output_dir).glob(f"{entity_type}_samples_*.jsonl"))
    if not existing_files:
        return f"{entity_type}_samples_01.jsonl"
    
    # Get the highest version number
    existing_versions = [
        int(file.stem.split('_')[-1]) for file in existing_files if file.stem.split('_')[-1].isdigit()
    ]
    next_version = max(existing_versions) + 1
    return f"{entity_type}_samples_{next_version:02}.jsonl"

def save_samples(sampled_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            doc_id = item.get("doc_id")
            text = item.get("text", "")
            sentences = preprocess_text(text)  # Split into individual sentences

            for sentence in sentences:
                # Each sentence is written as a separate JSON line with "text" key
                output_item = {
                    "doc_id": doc_id,
                    "text": sentence  # Use "text" for compatibility with Prodigy
                }
                f.write(json.dumps(output_item) + '\n')

def main():
    # Prompt the user for num_samples and entity_type
    entity_type = input("Enter the entity type (e.g., institution): ").strip()
    num_samples = int(input("Enter the number of samples to extract: "))

    # Ensure directories exist
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    log_file = os.path.join(LOG_DIR, f"{entity_type}_sampling_log.json")
    output_file = os.path.join(OUTPUT_DIR, get_next_filename(entity_type, OUTPUT_DIR))

    # Sample and preprocess data
    sampled_data = sample_data(num_samples, log_file)
    save_samples(sampled_data, output_file)

    print(f"Saved {num_samples} samples to {output_file}")

if __name__ == "__main__":
    main()
