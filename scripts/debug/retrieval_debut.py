import pandas as pd
import json
from pathlib import Path

# Load chunks data
chunks_path = Path(r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\RAG\embeddings\embeddings_1946_1946\chunks_embedding_index_country.csv")
chunks_df = pd.read_csv(chunks_path)
print("\nChunks DataFrame columns:")
print(chunks_df.columns.tolist())
print("\nSample of chunks data:")
print(chunks_df.head())

# Load target data
target_file = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\ungdc_model-v5_TargetContext_01.jsonl"
target_data = []
with open(target_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['year'] == 1946:
            target_data.append(data)

target_df = pd.DataFrame(target_data)
print("\nTarget DataFrame columns:")
print(target_df.columns.tolist())
print("\nSample of target data:")
print(target_df.head())

# Check what we're trying to match
print("\nUnique values in chunks_df['iso']:")
print(chunks_df['iso'].unique())

print("\nUnique values in target_df['target']:")
print(target_df['target'].unique())