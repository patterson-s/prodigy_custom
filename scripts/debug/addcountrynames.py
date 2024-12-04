import pandas as pd
from pathlib import Path

def enrich_chunks_with_country_names(
    chunks_file: str,
    iso_mapping_file: str,
    output_file: str = None
) -> pd.DataFrame:
    """
    Enrich chunks data with full country names based on ISO codes.
    """
    # Load chunks data
    print(f"Loading chunks from {chunks_file}")
    chunks_df = pd.read_csv(chunks_file)
    print("\nChunks DataFrame columns:")
    print(chunks_df.columns.tolist())
    print("\nFirst few rows of chunks data:")
    print(chunks_df.head())
    
    # Load ISO mapping data
    print(f"\nLoading ISO mappings from {iso_mapping_file}")
    iso_df = pd.read_json(iso_mapping_file, lines=True)
    print("\nISO mapping DataFrame columns:")
    print(iso_df.columns.tolist())
    print("\nFirst few rows of ISO mapping:")
    print(iso_df.head())
    
    print("\nBeginning merge operation...")
    print("Sample doc_id from chunks:", chunks_df['doc_id'].iloc[0])
    
    # Extract ISO code from doc_id (assuming format COUNTRY_NUMBER_YEAR)
    chunks_df['iso'] = chunks_df['doc_id'].str.split('_').str[0]
    print("\nAfter extracting ISO from doc_id:")
    print(chunks_df[['doc_id', 'iso']].head())
    
    # Merge country names based on ISO code
    print("\nMerging with ISO mappings...")
    chunks_df = chunks_df.merge(
        iso_df,
        on='iso',
        how='left'
    )
    
    # Create source/target columns
    chunks_df['source'] = chunks_df['iso']
    chunks_df['target'] = chunks_df['iso']
    chunks_df['source_country'] = chunks_df['country']
    chunks_df['target_country'] = chunks_df['country']
    
    # Arrange columns
    columns_to_keep = [
        'doc_id',
        'chunk_id',
        'year',
        'source',
        'source_country',
        'target',
        'target_country',
        'chunk_text',
        'chunk_start',
        'chunk_end',
        'original_text_length',
        'embedding_index'
    ]
    
    chunks_df = chunks_df[columns_to_keep]
    
    # Check for missing mappings
    missing_countries = chunks_df[chunks_df['source_country'].isna()]['source'].unique()
    
    if len(missing_countries) > 0:
        print("\nWarning: Missing country names for ISO codes:")
        print(missing_countries)
    
    # Save enriched data
    if output_file:
        print(f"\nSaving enriched data to {output_file}")
        chunks_df.to_csv(output_file, index=False)
        
    print("\nEnrichment complete!")
    print(f"Processed {len(chunks_df)} chunks")
    
    return chunks_df

if __name__ == "__main__":
    # Set up paths
    base_dir = Path(r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/RAG")
    
    chunks_file = base_dir / "embeddings/embeddings_1946_1946/chunks_with_embedding_index.csv"
    iso_mapping_file = Path(r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/patterns/iso_country.jsonl")
    output_file = base_dir / "embeddings/embeddings_1946_1946/chunks_with_countries.csv"
    
    # Create enriched dataset
    enriched_df = enrich_chunks_with_country_names(
        chunks_file=str(chunks_file),
        iso_mapping_file=str(iso_mapping_file),
        output_file=str(output_file)
    )