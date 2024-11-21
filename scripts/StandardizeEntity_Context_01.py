import pandas as pd
import nltk

# Ensure NLTK tokenizer is available
nltk.download('punkt')

# Function to extract context for each entity
def extract_context(row, token_window=10):
    """
    Extracts context for each entity in the 'gpe_entities' column from the 'text' column.
    Sequentially processes entities to ensure proper matching of multiple occurrences.
    """
    if pd.isna(row['gpe_entities']):
        return []  # Return an empty list if gpe_entities is NaN

    text_tokens = nltk.word_tokenize(row['text'])  # Tokenize the text
    gpe_entities = row['gpe_entities'].split(';')  # Split gpe_entities by ';'
    contexts = []
    used_indices = set()  # Keep track of processed indices
    tracked_pairs = set()  # Track unique doc_id + entity + context pairs to avoid duplicates

    for entity in gpe_entities:
        try:
            # Find the next occurrence of the entity not already processed
            index = next(
                i for i, token in enumerate(text_tokens)
                if token.lower() == entity.lower() and i not in used_indices
            )
            start = max(0, index - token_window)
            end = min(len(text_tokens), index + token_window + 1)
            context = ' '.join(text_tokens[start:end])

            # Avoid duplicates by checking if the pair already exists
            pair_key = (row['doc_id'], entity, context)
            if pair_key not in tracked_pairs:
                contexts.append({'entity': entity, 'context': context})
                tracked_pairs.add(pair_key)  # Mark this pair as seen

            # Mark the index as processed
            used_indices.add(index)
        except StopIteration:
            # Entity not found in remaining unprocessed tokens
            contexts.append({'entity': entity, 'context': None})

    return contexts

def main():
    # Step 1: Load the corpus with gpe_entities
    ungdc_path = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/ungdc_model-v5.csv"  # CORPUS FILE WITH ENTITIES
    ungdc = pd.read_csv(ungdc_path)

    # Step 2: Load the state_demonym_iso file
    state_demonym_iso_path = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/patterns/state_demonym_iso_01.jsonl"  # FILE WITH ISO key match
    state_demonym_iso = pd.read_json(state_demonym_iso_path, lines=True)

    # Step 3: Extract GPE contexts
    exploded_rows = []
    for _, row in ungdc.iterrows():
        row_contexts = extract_context(row)  # Extract contexts for all entities in the row
        for context_entry in row_contexts:
            # Create a new row for each entity-context pair
            exploded_rows.append({
                'doc_id': row['doc_id'],
                'iso': row['iso'],
                'session': row['session'],
                'year': row['year'],
                'gpe_entity': context_entry['entity'],
                'gpe_context': context_entry['context']
            })

    # Create a new dataframe from the exploded rows
    ungdc_gpeentity = pd.DataFrame(exploded_rows)

    # Step 4: Merge the ISO codes
    ungdc_gpeentity = ungdc_gpeentity.merge(
        state_demonym_iso[["pattern", "ISO_Code"]],
        left_on="gpe_entity",
        right_on="pattern",
        how="left"
    )
    ungdc_gpeentity.drop(columns=["pattern"], inplace=True)  # Drop the redundant 'pattern' column

    # Save the final dataframe to a CSV for reproducibility
    output_path = 'C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/ungdc_model-v5_EntityContext_01.csv'  # Replace with desired output path
    ungdc_gpeentity.to_csv(output_path, index=False)

    print(f"Processing complete. Output saved to {output_path}")

if __name__ == '__main__':
    main()
