import os
import pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from tqdm import tqdm

class UNGDCChunker:
    def __init__(self, 
                 input_path: str, 
                 output_path: str,
                 chunk_size: int = 512, 
                 chunk_overlap: int = 50):
        """Initialize the UNGDC document chunker.
        
        Args:
            input_path: Path to the raw UNGDC CSV file
            output_path: Path where the chunked data will be saved
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.input_path = Path(input_path)
        self.output_dir = Path(output_path)
        self.output_file = self.output_dir / 'ungdc_chunk.csv'
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def process_dataset(self):
        """Process the entire UNGDC dataset."""
        print("Loading UNGDC dataset...")
        df = pd.read_csv(self.input_path)
        
        # Keep only relevant columns
        df = df[['doc_id', 'iso', 'year', 'text']]
        
        # Initialize lists to store results
        all_chunks = []
        
        print("Processing speeches...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            doc_id = row['doc_id']  # Using original doc_id
            chunks_data = self.chunk_document(doc_id, row['text'])
            
            # Add each chunk as a row with metadata
            for chunk_idx, (chunk, position) in enumerate(zip(chunks_data['chunks'], 
                                                            chunks_data['positions'])):
                all_chunks.append({
                    'doc_id': doc_id,
                    'chunk_id': f"{doc_id}_chunk_{chunk_idx}",
                    'iso': row['iso'],
                    'year': row['year'],
                    'chunk_text': chunk,
                    'chunk_start': position['start'],
                    'chunk_end': position['end'],
                    'original_text_length': len(row['text'])
                })
        
        # Convert to DataFrame and save
        print("Saving chunked dataset...")
        chunks_df = pd.DataFrame(all_chunks)
        chunks_df.to_csv(self.output_file, index=False)
        print(f"Saved {len(chunks_df)} chunks to {self.output_file}")
        
        # Print some statistics
        print("\nChunking Statistics:")
        print(f"Total documents processed: {len(df)}")
        print(f"Total chunks created: {len(chunks_df)}")
        print(f"Average chunks per document: {len(chunks_df)/len(df):.2f}")
        
        return chunks_df
    
    def chunk_document(self, doc_id: str, text: str) -> Dict[str, List]:
        """Split a document into chunks and track their positions."""
        # Split text into chunks
        chunks = self._create_chunks(text)
        
        # Track chunk positions in original text
        positions = self._get_chunk_positions(text, chunks)
        
        return {
            'doc_id': doc_id,
            'chunks': chunks,
            'positions': positions
        }
    
    def _create_chunks(self, text: str) -> List[str]:
        """Split text into chunks using LangChain splitter."""
        docs = self.text_splitter.create_documents([text])
        return [doc.page_content for doc in docs]
    
    def _get_chunk_positions(self, text: str, chunks: List[str]) -> List[Dict[str, int]]:
        """Track the position of each chunk in the original text."""
        positions = []
        current_pos = 0
        
        for chunk in chunks:
            chunk_start = text.find(chunk, current_pos)
            positions.append({
                'start': chunk_start,
                'end': chunk_start + len(chunk)
            })
            current_pos = chunk_start + 1
            
        return positions

if __name__ == "__main__":
    # Set up paths
    input_path = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\raw\ungdc_1946-2022.csv"
    output_path = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\RAG"
    
    # Create and run the chunker
    chunker = UNGDCChunker(input_path, output_path)
    chunks_df = chunker.process_dataset()
