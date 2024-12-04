import spacy
import pandas as pd
from pathlib import Path

# Paths
MODEL_PATH = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/models/model-v5/model-best"
INPUT_FILE = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/RAG/embeddings/embeddings_1946_1946/chunks_embedding_index_country.csv"
OUTPUT_FILE = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/ungdc_model-v5_chunk.csv"
ISO_MAPPING_FILE = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/patterns/iso_match_master_01.jsonl"

def load_iso_mapping(mapping_file):
    """Load ISO code mapping from JSONL file."""
    print("Loading ISO mapping...")
    mapping_df = pd.read_json(mapping_file, lines=True)
    # Create dictionary for faster lookups
    return dict(zip(mapping_df['pattern'], mapping_df['ISO_Code']))

def convert_to_iso_codes(entities, iso_mapping):
    """Convert a string of semicolon-separated entities to ISO codes."""
    if pd.isna(entities):
        return None
    
    # Split entities and convert each one
    entity_list = entities.split(';')
    iso_codes = []
    
    for entity in entity_list:
        entity = entity.strip()
        iso_code = iso_mapping.get(entity)
        if iso_code:
            iso_codes.append(iso_code)
        else:
            # If no mapping found, keep original entity for review
            iso_codes.append(f"UNKNOWN_{entity}")
    
    # Join back with semicolons
    return ';'.join(iso_codes)

def process_with_model(input_path, model_path, iso_mapping_path, output_path):
    # Load the trained model
    print("Loading model...")
    nlp = spacy.load(model_path)
    
    # Load ISO mapping
    iso_mapping = load_iso_mapping(iso_mapping_path)
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    if 'chunk_text' not in df.columns:
        raise ValueError("Dataset must have a 'chunk_text' column containing the speeches.")
    
    # Process texts and extract GPEs
    gpe_entities = []
    print("Processing dataset with model...")
    for doc_text in df['chunk_text']:
        doc = nlp(doc_text)
        gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        gpe_entities.append(";".join(gpes))  # Join entities with ';'
    
    # Add the GPEs as a new column
    df['gpe_entities'] = gpe_entities
    
    # Convert GPE entities to ISO codes
    print("Converting entities to ISO codes...")
    df['target_iso'] = df['gpe_entities'].apply(
        lambda x: convert_to_iso_codes(x, iso_mapping)
    )
    
    # Save the updated dataset
    print(f"Saving updated dataset to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Processing complete.")

if __name__ == "__main__":
    process_with_model(INPUT_FILE, MODEL_PATH, ISO_MAPPING_FILE, OUTPUT_FILE)
