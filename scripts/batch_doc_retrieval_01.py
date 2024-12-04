import os
import json
import numpy as np
import pandas as pd
import cohere
from pathlib import Path
from typing import List, Dict, Optional, Set
from tqdm import tqdm
import time

class BatchDocumentRetriever:
    def __init__(self, cohere_api_key: str, output_dir: str, iso_mapping_file: str):
        """Initialize batch document retrieval system.
        
        Args:
            cohere_api_key: API key for Cohere
            output_dir: Directory to save results
            iso_mapping_file: Path to ISO to country name mapping file
        """
        self.co = cohere.Client(cohere_api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load ISO to country mapping
        self.iso_mapping = {}
        with open(iso_mapping_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.iso_mapping[data['iso']] = data['country']
        
        # Initialize tracking variables
        self.current_doc_id: Optional[str] = None
        self.chunks_df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None
        self.target_data: Optional[Dict] = None
        
        # Add processing stats
        self.stats = {
            'total_documents': 0,
            'successful_documents': 0,
            'failed_documents': 0,
            'total_targets_processed': 0,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def load_target_metadata(self, target_file: str) -> Dict[str, Set[str]]:
        """Load target country data from JSONL file."""
        doc_targets = {}
        with open(target_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data['year'] == 1946:  # Filter for 1946
                    doc_id = data['doc_id']
                    if doc_id not in doc_targets:
                        doc_targets[doc_id] = set()
                    doc_targets[doc_id].add(data['target'])
        return doc_targets

    def get_country_name(self, iso_code: str) -> Optional[str]:
        """Get country name from ISO code using mapping file."""
        return self.iso_mapping.get(iso_code)

    def construct_retrieval_query(self, target_iso: str, target_country: str) -> str:
        """Construct query to find mentions of target country."""
        return f"mentions discussions or references to {target_iso} OR {target_country}"

    def retrieve_chunks_for_target(self, 
                                 target_iso: str, 
                                 target_country: str, 
                                 top_k: int = 10) -> List[Dict]:
        """Retrieve chunks mentioning specific target country."""
        if self.current_doc_id is None:
            raise ValueError("No document currently loaded")
            
        try:
            # Construct and embed query
            query = self.construct_retrieval_query(target_iso, target_country)
            query_response = self.co.embed(
                texts=[query],
                model="embed-english-v3.0",
                input_type="search_query"
            )
            
            # Handle the embedding response correctly
            if hasattr(query_response.embeddings, 'float'):
                query_embedding = np.array(query_response.embeddings.float[0])
            else:
                query_embedding = np.array(query_response.embeddings[0])
            
            # Calculate similarities
            similarities = np.dot(self.embeddings, query_embedding)
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Get top chunks with metadata
            results = []
            for idx in top_indices:
                chunk_data = self.chunks_df.iloc[idx].to_dict()
                results.append({
                    'chunk_text': chunk_data['chunk_text'],
                    'similarity': float(similarities[idx]),
                    'position': {
                        'start': chunk_data['chunk_start'],
                        'end': chunk_data['chunk_end']
                    },
                    'metadata': {
                        'doc_id': self.current_doc_id,
                        'chunk_id': chunk_data['chunk_id'],
                        'embedding_index': chunk_data['embedding_index']
                    }
                })
                
            return results
            
        except Exception as e:
            print(f"Error in retrieve_chunks_for_target: {str(e)}")
            return []

    def retrieve_all_targets(self, top_k: int = 10, top_n_rerank: int = 5) -> Dict[str, List[Dict]]:
        """Retrieve chunks for all target countries in current document."""
        if self.current_doc_id is None:
            raise ValueError("No document currently loaded")
            
        results = {}
        target_countries = self.target_data[self.current_doc_id]
        
        for target_iso in target_countries:
            try:
                if pd.isna(target_iso):  # Skip if target is NaN
                    print(f"Skipping invalid target ISO: {target_iso}")
                    continue
                    
                print(f"\nProcessing target country: {target_iso}")
                
                # Get country name from ISO mapping
                country_name = self.get_country_name(target_iso)
                if not country_name:
                    print(f"No country name found for target: {target_iso}")
                    continue
                
                print(f"Found country name: {country_name} for target: {target_iso}")
                
                # Get initial chunks
                target_chunks = self.retrieve_chunks_for_target(
                    target_iso=target_iso,
                    target_country=country_name,
                    top_k=top_k
                )
                
                # Rerank chunks if we got any
                if target_chunks:
                    try:
                        reranked = self.rerank_chunks(
                            chunks=target_chunks,
                            target_iso=target_iso,
                            target_country=country_name,
                            top_n=top_n_rerank
                        )
                        if reranked:  # Only store if reranking succeeded
                            results[target_iso] = reranked
                    except Exception as e:
                        print(f"Error during reranking for {target_iso}: {str(e)}")
                        # Fall back to non-reranked results
                        results[target_iso] = target_chunks[:top_n_rerank]
                else:
                    print(f"No chunks found for target {target_iso}")
            
            except Exception as e:
                print(f"Error processing target {target_iso}: {str(e)}")
                continue
            
        return results

    def rerank_chunks(self, 
                     chunks: List[Dict], 
                     target_iso: str,
                     target_country: str,
                     top_n: int = 3) -> List[Dict]:
        """Rerank retrieved chunks using Cohere's rerank endpoint."""
        query = self.construct_retrieval_query(target_iso, target_country)
        texts = [chunk['chunk_text'] for chunk in chunks]
        
        rerank_response = self.co.rerank(
            query=query,
            documents=texts,
            top_n=top_n,
            model="rerank-english-v2.0"
        )
        
        reranked = []
        for result in rerank_response:
            try:
                # Handle different response formats
                if hasattr(result, 'document'):
                    text = result.document['text']
                else:
                    text = result.text
                
                original_idx = texts.index(text)
                reranked.append({
                    **chunks[original_idx],
                    'rerank_score': result.relevance_score
                })
            except Exception as e:
                print(f"Error processing rerank result: {str(e)}")
                continue
            
        return reranked

    def process_documents(self, 
                        embeddings_dir: Path,
                        target_file: str,
                        print_frequency: int = 5):
        """Process all 1946 documents."""
        try:
            # Load target data once
            self.target_data = self.load_target_metadata(target_file)
            print(f"Loaded target data for {len(self.target_data)} documents from 1946")
            
            # Load all chunks and embeddings once
            chunks_path = embeddings_dir / 'chunks_embedding_index_country.csv'
            self.all_chunks_df = pd.read_csv(chunks_path)
            self.all_embeddings = np.load(embeddings_dir / 'embeddings.npy')
            
            # Debug info
            print(f"\nChunks data columns: {self.all_chunks_df.columns.tolist()}")
            
            # Get unique document IDs
            doc_ids = sorted(list(self.target_data.keys()))
            self.stats['total_documents'] = len(doc_ids)
            
            print(f"\nProcessing {len(doc_ids)} documents from 1946...")
            
            # Process each document
            for idx, doc_id in enumerate(tqdm(doc_ids), 1):
                try:
                    # Filter chunks for this document
                    self.chunks_df = self.all_chunks_df[self.all_chunks_df['doc_id'] == doc_id].copy()
                    if len(self.chunks_df) == 0:
                        print(f"\nSkipping {doc_id}: No chunks found")
                        self.stats['failed_documents'] += 1
                        continue
                    
                    # Get embeddings for this document's chunks
                    doc_embedding_indices = self.chunks_df['embedding_index'].values
                    self.embeddings = self.all_embeddings[doc_embedding_indices]
                    
                    # Set current document
                    self.current_doc_id = doc_id
                    
                    # Process all targets for this document
                    results = self.retrieve_all_targets()
                    
                    # Only save if we got results
                    if results:
                        output_file = self.output_dir / f"{doc_id}_retrieval.json"
                        with open(output_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                        self.stats['successful_documents'] += 1
                        self.stats['total_targets_processed'] += len(results)
                    else:
                        print(f"\nNo results found for document {doc_id}")
                        self.stats['failed_documents'] += 1
                    
                    # Print progress update
                    if idx % print_frequency == 0:
                        print(f"\nProgress Update - Document {idx}/{len(doc_ids)}")
                        print(f"Current document: {doc_id}")
                        print(f"Number of targets: {len(results)}")
                        for target, chunks in results.items():
                            print(f"- {target}: {len(chunks)} chunks retrieved")
                
                except Exception as e:
                    print(f"\nError processing document {doc_id}: {str(e)}")
                    self.stats['failed_documents'] += 1
            
            # Save final stats
            self.stats['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(self.output_dir / 'processing_stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
            
            print("\nProcessing complete!")
            print(f"Successful documents: {self.stats['successful_documents']}")
            print(f"Failed documents: {self.stats['failed_documents']}")
            print(f"Total targets processed: {self.stats['total_targets_processed']}")
            print(f"Results saved in: {self.output_dir}")
            
        except Exception as e:
            print(f"Critical error in process_documents: {str(e)}")
            raise

if __name__ == "__main__":
    # Get API key
    api_key = os.getenv('COHERE_API_KEY')
    if not api_key:
        raise ValueError("COHERE_API_KEY environment variable not found")
    
    # Set up paths
    base_dir = Path(r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/RAG")
    embeddings_dir = base_dir / "embeddings/embeddings_1946_1946"
    target_file = r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/ungdc_model-v5_TargetContext_01.jsonl"
    iso_mapping_file = r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/patterns/iso_country.jsonl"
    output_dir = r"C:/Users/spatt/Desktop/diss_3/prodigy_custom/data/processed/RAG\retrieval"
    
    # Initialize and run batch processor
    retriever = BatchDocumentRetriever(api_key, output_dir, iso_mapping_file)
    retriever.process_documents(
        embeddings_dir=embeddings_dir,
        target_file=target_file,
        print_frequency=5
    )