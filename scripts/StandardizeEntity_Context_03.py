import pandas as pd
import nltk

# Ensure NLTK tokenizer is available
nltk.download('punkt')

# Function to extract context for each entity
def extract_context(row, token_window=100):
    if pd.isna(row['gpe_entities']):
        return []

    text_tokens = nltk.word_tokenize(row['text'])  # Tokenize the text
    gpe_entities = row['gpe_entities'].split(';')  # Split entities by `;`
    contexts = []
    used_indices = set()
    tracked_pairs = set()

    for entity in gpe_entities:
        try:
            index = next(
                i for i, token in enumerate(text_tokens)
                if token.lower() == entity.lower() and i not in used_indices
            )
            start = max(0, index - token_window)
            end = min(len(text_tokens), index + token_window + 1)
            context = ' '.join(text_tokens[start:end])

            pair_key = (row['doc_id'], entity, context)
            if pair_key not in tracked_pairs:
                contexts.append({'entity': entity, 'context': context})
                tracked_pairs.add(pair_key)
            used_indices.add(index)
        except StopIteration:
            contexts.append({'entity': entity, 'context': None})

    return contexts

def main():
    # Updated file paths
    ungdc_path = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/ungdc_chunk_gpe.csv"
    state_demonym_iso_path = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/patterns/iso_match_master_01.jsonl"
    drop_patterns_path = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/patterns/iso_drop_master_01.jsonl"
    output_path = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/ungdc_chunk_model-v5_EntityContext.csv"

    # Load datasets
    ungdc = pd.read_csv(ungdc_path)
    state_demonym_iso = pd.read_json(state_demonym_iso_path, lines=True)
    drop_patterns = pd.read_json(drop_patterns_path, lines=True)

    # Extract patterns to drop
    drop_patterns_list = drop_patterns['pattern'].str.lower().tolist()

    # Extract contexts
    exploded_rows = []
    for _, row in ungdc.iterrows():
        row_contexts = extract_context(row)
        for context_entry in row_contexts:
            exploded_rows.append({
                'doc_id': row['doc_id'],
                'chunk_id': row['chunk_id'],
                'iso': row['iso'],
                'year': row['year'],
                'chunk_start': row['chunk_start'],
                'chunk_end': row['chunk_end'],
                'original_text_length': row['original_text_length'],
                'text': row['text'],  # Added this line to keep the text column
                'gpe_entity': context_entry['entity'],
                'gpe_context': context_entry['context']
            })
    ungdc_gpeentity = pd.DataFrame(exploded_rows)
    print(f"Entities after context extraction: {len(ungdc_gpeentity)}")

    # Merge with ISO codes
    ungdc_gpeentity = ungdc_gpeentity.merge(
        state_demonym_iso[["pattern", "ISO_Code"]],
        left_on="gpe_entity",
        right_on="pattern",
        how="left"
    )
    ungdc_gpeentity.drop(columns=["pattern"], inplace=True)
    print(f"Entities after merging ISO codes: {len(ungdc_gpeentity)}")

    # Case-insensitive filtering
    ungdc_gpeentity['gpe_entity_lower'] = ungdc_gpeentity['gpe_entity'].str.lower()
    pre_filter_count = len(ungdc_gpeentity)
    ungdc_gpeentity = ungdc_gpeentity[
        ~ungdc_gpeentity['gpe_entity_lower'].isin(drop_patterns_list)
    ]
    post_filter_count = len(ungdc_gpeentity)
    print(f"Entities before filtering: {pre_filter_count}, after filtering: {post_filter_count}")

    # Drop temporary columns
    ungdc_gpeentity.drop(columns=["gpe_entity_lower"], inplace=True)

    # Save final results
    ungdc_gpeentity.to_csv(output_path, index=False)
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == '__main__':
    main()