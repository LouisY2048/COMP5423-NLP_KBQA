"""
Script to train dense retriever model and build FAISS index using training data
"""

import os
import json
from typing import List, Dict, Any
from tqdm import tqdm
import sys

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

sys.path.append(project_root+"/NLP_KBQA")

from dense_retrieval import DenseRetriever

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from jsonl file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def prepare_documents(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prepare documents for indexing"""
    documents = []
    for item in data:
        doc = {
            'document_id': item['document_id'],
            'document_text': item['document_text']  # Using document text
        }
        documents.append(doc)
    return documents

def main():
    # File paths
    train_path = os.path.join(project_root, "data", "train.jsonl")
    docs_path = os.path.join(project_root, "data", "documents.jsonl")
    model_dir = os.path.join(project_root, "backend", "models", "dense_retriever")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load documents and training data
    print("Loading documents and training data...")
    documents = load_jsonl(docs_path)
    train_data = load_jsonl(train_path)
    
    print(f"Loaded {len(documents)} documents and {len(train_data)} training examples")
    
    # Initialize retriever
    print("Initializing DenseRetriever...")
    retriever = DenseRetriever(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        index_type='Flat',
        device='cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'
    )
    
    # Prepare documents
    print("Preparing documents...")
    doc_data = prepare_documents(documents)
    
    # Build index and train
    print("Building index and training...")
    retriever.build_index(doc_data, train_data=train_data)
    
    # Save model and index
    print(f"Saving model and index to {model_dir}")
    retriever.save_index(model_dir)
    print("Done!")

if __name__ == "__main__":
    main()