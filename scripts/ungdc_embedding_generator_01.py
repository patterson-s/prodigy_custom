import os
import pandas as pd
import numpy as np
from pathlib import Path
import cohere
from tqdm import tqdm
import json
from datetime import datetime
import time
from typing import List, Dict, Any
import logging

class UNGDCEmbeddingsGenerator:
    def __init__(self, 
                 chunks_path: str,
                 output_path: str,
                 cohere_api_key: str,
                 batch_size: int = 96,
                 max_retries: int = 3,
                 checkpoint_frequency: int = 10):
        """Initialize the UNGDC embeddings generator.
        
        Args:
            chunks_path: Path to the chunked CSV file
            output_path: Path where embeddings will be saved
            cohere_api_key: Cohere API key
            batch_size: Number of texts to embed in each API call
            max_retries: Maximum number of retry attempts for failed API calls
            checkpoint_frequency: Save checkpoint every N batches
        """
        self.chunks_path = Path(chunks_path)
        self.output_dir = Path(output_path)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.checkpoint_frequency = checkpoint_frequency
        self.co = cohere.Client(cohere_api_key)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize usage tracking
        self.usage_stats = {
            'total_chunks': 0,
            'total_tokens': 0,
            'total_api_calls': 0,
            'failed_calls': 0,
            'retried_calls': 0,
            'start_time': datetime.now().isoformat(),
        }
    
    def _generate_batch_embeddings(self, batch: List[str], retry_count: int = 0) -> List[List[float]]:
        """Generate embeddings for a batch with retry logic."""
        try:
            response = self.co.embed(
                texts=batch,
                model="embed-english-v3.0",
                input_type="search_document",
                embedding_types=['float']
            )
            return response.embeddings.float
            
        except cohere.error.CohereError as e:
            self.usage_stats['failed_calls'] += 1
            
            if "rate_limit" in str(e).lower() and retry_count < self.max_retries:
                self.logger.warning(f"Rate limit hit, waiting before retry {retry_count + 1}/{self.max_retries}")
                time.sleep(20 * (retry_count + 1))  # Exponential backoff
                self.usage_stats['retried_calls'] += 1
                return self._generate_batch_embeddings(batch, retry_count + 1)
            else:
                self.logger.error(f"Embedding generation failed: {str(e)}")
                raise
    
    def _save_checkpoint(self, 
                        year_dir: Path, 
                        embeddings: List[List[float]], 
                        processed_chunks: pd.DataFrame,
                        batch_num: int):
        """Save intermediate results checkpoint."""
        checkpoint_dir = year_dir / f"checkpoint_{batch_num}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save embeddings
        embeddings_array = np.array(embeddings)
        np.save(checkpoint_dir / 'embeddings_checkpoint.npy', embeddings_array)
        
        # Save processed chunks
        processed_chunks.to_csv(checkpoint_dir / 'chunks_checkpoint.csv', index=False)
        
        # Save usage stats
        with open(checkpoint_dir / 'usage_stats_checkpoint.json', 'w') as f:
            json.dump(self.usage_stats, f, indent=2)
        
        self.logger.info(f"Saved checkpoint {batch_num} with {len(embeddings)} embeddings")
    
    def _load_latest_checkpoint(self, year_dir: Path) -> tuple:
        """Load the latest checkpoint if it exists."""
        checkpoints = list(year_dir.glob("checkpoint_*"))
        if not checkpoints:
            return None, None, None
        
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('_')[1]))
        self.logger.info(f"Found checkpoint: {latest_checkpoint}")
        
        # Load embeddings
        embeddings = np.load(latest_checkpoint / 'embeddings_checkpoint.npy').tolist()
        
        # Load processed chunks
        chunks_df = pd.read_csv(latest_checkpoint / 'chunks_checkpoint.csv')
        
        # Load usage stats
        with open(latest_checkpoint / 'usage_stats_checkpoint.json', 'r') as f:
            usage_stats = json.load(f)
        
        return embeddings, chunks_df, usage_stats
    
    def generate_embeddings(self, start_year: int, end_year: int):
        """Generate embeddings for chunks from specified years."""
        self.logger.info(f"Loading chunks for years {start_year}-{end_year}...")
        df = pd.read_csv(self.chunks_path)
        
        # Filter for specified years
        year_mask = (df['year'] >= start_year) & (df['year'] <= end_year)
        df_filtered = df[year_mask].copy()
        
        self.logger.info(f"Found {len(df_filtered)} chunks from {len(df_filtered['doc_id'].unique())} speeches")
        
        # Create output subdirectory for this year range
        year_dir = self.output_dir / f"embeddings_{start_year}_{end_year}"
        year_dir.mkdir(exist_ok=True)
        
        # Check for existing checkpoint
        checkpoint_data = self._load_latest_checkpoint(year_dir)
        if checkpoint_data != (None, None, None):
            all_embeddings, df_filtered, self.usage_stats = checkpoint_data
            start_idx = len(all_embeddings)
            self.logger.info(f"Resuming from checkpoint with {start_idx} embeddings")
        else:
            all_embeddings = []
            start_idx = 0
        
        # Process in batches
        chunks = df_filtered['chunk_text'].tolist()[start_idx:]
        total_batches = len(chunks) // self.batch_size + (1 if len(chunks) % self.batch_size else 0)
        
        self.logger.info("Generating embeddings...")
        try:
            for batch_num, i in enumerate(tqdm(range(0, len(chunks), self.batch_size)), start=start_idx//self.batch_size):
                batch = chunks[i:i + self.batch_size]
                
                # Generate embeddings for batch
                batch_embeddings = self._generate_batch_embeddings(batch)
                
                # Update usage statistics
                self.usage_stats['total_chunks'] += len(batch)
                self.usage_stats['total_api_calls'] += 1
                
                # Store embeddings
                all_embeddings.extend(batch_embeddings)
                
                # Save checkpoint if needed
                if batch_num % self.checkpoint_frequency == 0:
                    self._save_checkpoint(
                        year_dir, 
                        all_embeddings, 
                        df_filtered.iloc[:len(all_embeddings)],
                        batch_num
                    )
        
        except Exception as e:
            self.logger.error(f"Error during embedding generation: {str(e)}")
            # Save final checkpoint before exiting
            self._save_checkpoint(
                year_dir, 
                all_embeddings, 
                df_filtered.iloc[:len(all_embeddings)],
                'final_error'
            )
            raise
        
        # Convert to numpy array and save final results
        embeddings_array = np.array(all_embeddings)
        
        # Save final embeddings
        np.save(year_dir / 'embeddings.npy', embeddings_array)
        
        # Save metadata
        metadata = {
            'num_chunks': len(df_filtered),
            'num_speeches': len(df_filtered['doc_id'].unique()),
            'years': f"{start_year}-{end_year}",
            'embedding_model': "embed-english-v3.0",
            'embedding_dim': embeddings_array.shape[1],
            'created_at': datetime.now().isoformat()
        }
        
        with open(year_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save chunk information with embedding indices
        df_filtered['embedding_index'] = range(len(df_filtered))
        df_filtered.to_csv(year_dir / 'chunks_with_embedding_index.csv', index=False)
        
        # Update and save final usage statistics
        self.usage_stats['end_time'] = datetime.now().isoformat()
        with open(year_dir / 'usage_stats.json', 'w') as f:
            json.dump(self.usage_stats, f, indent=2)
        
        self.logger.info(f"\nEmbeddings generation complete!")
        self.logger.info(f"Created {len(df_filtered)} embeddings")
        self.logger.info(f"Files saved in: {year_dir}")
        self.logger.info(f"Total API calls: {self.usage_stats['total_api_calls']}")
        self.logger.info(f"Failed calls: {self.usage_stats['failed_calls']}")
        self.logger.info(f"Retried calls: {self.usage_stats['retried_calls']}")
        
        return metadata

if __name__ == "__main__":
    # Get API key from environment variable
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")

    # Interactive input for years
    while True:
        try:
            start_year = int(input("Enter start year: "))
            end_year = int(input("Enter end year: "))
            if 1946 <= start_year <= 2022 and 1946 <= end_year <= 2022 and start_year <= end_year:
                break
            else:
                print("Years must be between 1946 and 2022, and start year must not be later than end year.")
        except ValueError:
            print("Please enter valid years (numbers between 1946 and 2022)")
    
    # Confirm the year range with user
    print(f"\nWill process speeches from {start_year} to {end_year}")
    confirm = input("Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled")
        exit()
    
    # Set up paths
    chunks_path = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\RAG\ungdc_chunk.csv"
    output_path = r"C:\Users\spatt\Desktop\diss_3\prodigy_custom\data\processed\RAG\embeddings"
    
    # Create and run the embeddings generator
    generator = UNGDCEmbeddingsGenerator(
        chunks_path=chunks_path,
        output_path=output_path,
        cohere_api_key=api_key
    )
    
    metadata = generator.generate_embeddings(start_year, end_year)
