import spacy
import pandas as pd
from pathlib import Path

# Paths
MODEL_PATH = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/models/model-v5/model-best"
INPUT_FILE = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/raw/ungdc_1946-2022.csv"
OUTPUT_FILE = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/ungdc_model-v5.csv"

def process_with_model(input_path, model_path, output_path):
    # Load the trained model
    print("Loading model...")
    nlp = spacy.load(model_path)

    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    if 'text' not in df.columns:
        raise ValueError("Dataset must have a 'text' column containing the speeches.")
    
    # Process texts and extract GPEs
    gpe_entities = []
    print("Processing dataset with model...")
    for doc_text in df['text']:
        doc = nlp(doc_text)
        gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        gpe_entities.append(";".join(gpes))  # Join entities with ';'

    # Add the GPEs as a new column
    df['gpe_entities'] = gpe_entities

    # Save the updated dataset
    print(f"Saving updated dataset to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Processing complete.")

if __name__ == "__main__":
    process_with_model(INPUT_FILE, MODEL_PATH, OUTPUT_FILE)
