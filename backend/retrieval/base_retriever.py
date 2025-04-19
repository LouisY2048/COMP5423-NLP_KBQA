"""
Base Retriever module for the Knowledge Base Question Answering System.
This module provides the abstract base class for all retriever implementations.
It defines the common interface and functionality for document retrieval.

Key Features:
1. Abstract base class for document retrieval
2. Common configuration parameters (top_k, chunk_size, chunk_overlap)
3. Basic document scoring and retrieval interface
4. Model saving and loading functionality
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import os
import json

class BaseRetriever(ABC):
    """Abstract base class for all retrievers"""
    
    def __init__(self, top_k=5, chunk_size=50, chunk_overlap=20):
        """Initialize the retriever"""
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        """Retrieve relevant documents for the query"""
        doc_scores = self._get_doc_scores(query)

        top_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top_doc_ids = [doc_id for doc_id, _ in top_doc_ids]
        result_doc_ids = []
        result_doc_texts = []
        
        for doc_id in top_doc_ids:
            result_doc_ids.append(doc_id)
        
        return result_doc_ids, result_doc_texts
    
    def _get_doc_scores(self, question):
        """Get the relevance score of each document to the question (implemented by subclasses)"""
        raise NotImplementedError
    
    def _get_chunk_scores(self, question, chunks):
        """Get the relevance score of each chunk to the question (implemented by subclasses)"""
        raise NotImplementedError
    
    def save_model(self, path):
        """Save the model (implemented by subclasses)"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        config = {
            'top_k': self.top_k,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'total_chunks': len(self.chunks)
        }
        
        chunks_info = {
            'doc_ids': self.doc_ids,
            'chunks': self.chunks,
            'chunk_doc_ids': self.chunk_doc_ids,
            'chunk_indices': self.chunk_indices,
            'doc_to_chunks': self.doc_to_chunks
        }
        
        return config, chunks_info
    
    def _load_saved_data(self, path):
        """Load the saved data (implemented by subclasses)"""
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.chunk_size = config['chunk_size']
                self.chunk_overlap = config['chunk_overlap']
                self.top_k = config['top_k']
        
        chunks_path = os.path.join(path, 'chunks_info.json')
        if os.path.exists(chunks_path):
            with open(chunks_path, 'r') as f:
                chunks_info = json.load(f)
                self.doc_ids = chunks_info['doc_ids']
                self.chunks = chunks_info['chunks']
                self.chunk_doc_ids = chunks_info['chunk_doc_ids']
                self.chunk_indices = chunks_info['chunk_indices']
                self.doc_to_chunks = {int(k): v for k, v in chunks_info['doc_to_chunks'].items()} 