import pandas as pd
from pathlib import Path

# Paths
INPUT_FILE = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/ungdc_model-v5_chunk.csv"
OUTPUT_FILE = "C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/ungdc_model-v5_chunk_long.csv"

def lengthen_country_mentions(input_path, output_path):
    """
    Create separate rows for each country mention in the dataset.
    
    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the lengthened CSV file
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Initial shape:", df.shape)
    
    # Split the semicolon-separated values into lists
    print("Lengthening dataframe for individual country mentions...")
    df['gpe_entities'] = df['gpe_entities'].str.split(';')
    df['target_iso'] = df['target_iso'].str.split(';')
    
    # Explode both columns simultaneously
    lengthened_df = df.explode(['gpe_entities', 'target_iso'])
    
    # Reset index
    lengthened_df = lengthened_df.reset_index(drop=True)
    
    print("Final shape:", lengthened_df.shape)
    
    # Save the lengthened dataset
    print(f"Saving lengthened dataset to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lengthened_df.to_csv(output_path, index=False)
    
    # Print summary statistics
    total_mentions = len(lengthened_df)
    unique_countries = len(lengthened_df['target_iso'].unique())
    unique_chunks = len(lengthened_df['chunk_id'].unique())
    avg_mentions = total_mentions / unique_chunks
    
    print("\nSummary Statistics:")
    print(f"Total country mentions: {total_mentions:,}")
    print(f"Number of unique countries: {unique_countries}")
    print(f"Number of unique chunks: {unique_chunks:,}")
    print(f"Average mentions per chunk: {avg_mentions:.2f}")

if __name__ == "__main__":
    lengthen_country_mentions(INPUT_FILE, OUTPUT_FILE)