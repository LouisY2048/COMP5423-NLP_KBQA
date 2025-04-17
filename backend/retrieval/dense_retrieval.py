"""
Dense retrieval implementation using FAISS for efficient similarity search.
This module provides functionality to encode documents and queries into dense vectors
and perform approximate nearest neighbor search.
"""

import os
import faiss
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from .base_retriever import BaseRetriever
from tqdm import tqdm

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TrainDataset(Dataset):
    """Dataset for training dense retriever"""
    def __init__(self, questions: List[str], doc_ids: List[int]):
        self.questions = questions
        self.doc_ids = doc_ids
        
    def __len__(self):
        return len(self.questions)
        
    def __getitem__(self, idx):
        return self.questions[idx], self.doc_ids[idx]

class DenseRetriever(BaseRetriever):
    """Dense retriever using FAISS for efficient similarity search"""
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 index_type: str = 'Flat',
                 device: str = 'cpu',
                 **kwargs):
        """
        Initialize the dense retriever
        
        Args:
            model_name: Name/path of the sentence transformer model to use
            index_type: Type of FAISS index to use ('Flat' or 'IVF')
            device: Device to use for encoding ('cpu' or 'cuda')
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model_name = model_name
        self.index_type = index_type
        self.device = device
        
        # Load sentence transformer model
        self.model = SentenceTransformer(model_name)
        if device == 'cuda':
            self.model = self.model.to('cuda')
            
        self.index = None
        self.doc_ids = []
        self.doc_texts = []
        self.doc_embeddings = None
        
    def build_index(self, documents: List[Dict[str, Any]], train_data: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Build FAISS index from documents and optionally train with training data
        
        Args:
            documents: List of document dictionaries with 'document_id' and 'document_text' fields
            train_data: Optional list of training examples with 'question' and 'document_id' fields
        """
        # Extract document texts and IDs
        self.doc_texts = [doc['document_text'] for doc in documents]
        self.doc_ids = [doc['document_id'] for doc in documents]
        
        # Encode documents
        print("Encoding documents...")
        self.doc_embeddings = self.encode_batch(self.doc_texts)
        np.save(os.path.join(project_dir, 'models', 'dense_retriever', 'doc_embeddings.npy'), self.doc_embeddings)
        
        # Initialize FAISS index
        vector_dimension = self.doc_embeddings.shape[1]
        if self.index_type == 'Flat':
            self.index = faiss.IndexFlatIP(vector_dimension)
        elif self.index_type == 'IVF':
            nlist = min(len(documents) // 10, 100)  # Number of clusters
            quantizer = faiss.IndexFlatIP(vector_dimension)
            self.index = faiss.IndexIVFFlat(quantizer, vector_dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.doc_embeddings)
        
        # Add vectors to index
        self.index.add(self.doc_embeddings)
        
        # Train if training data is provided
        if train_data:
            self.train(train_data)
    
    def train(self, train_data: List[Dict[str, Any]], 
              batch_size: int = 32,
              epochs: int = 3,
              learning_rate: float = 2e-5,
              num_negatives: int = 5) -> None:
        """
        Train the model using training data
        
        Args:
            train_data: List of training examples with 'question' and 'document_id' fields
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate for fine-tuning
            num_negatives: Number of negative samples per positive example
        """
        print("Starting training...")
        
        # Prepare training data
        questions = [item['question'] for item in train_data]
        doc_ids = [item['document_id'] for item in train_data]
        
        # Create dataset and dataloader
        dataset = TrainDataset(questions, doc_ids)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Move model to appropriate device and set to training mode
        self.model.to(self.device)
        self.model.train()
        
        # Prepare optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Convert document embeddings to tensor once
        doc_embeddings_tensor = torch.tensor(self.doc_embeddings, device=self.device)
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            # Create progress bar for this epoch
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', 
                       unit='batch', position=0, leave=True)
            
            for batch_questions, batch_doc_ids in pbar:
                optimizer.zero_grad()
                
                # Get question embeddings with gradients
                question_embeddings = self.encode_batch(batch_questions, is_training=True)
                
                # Calculate similarity scores using matrix multiplication
                similarity_scores = torch.matmul(question_embeddings, doc_embeddings_tensor.T)
                
                # Create target labels
                targets = torch.tensor([self.doc_ids.index(doc_id) for doc_id in batch_doc_ids],
                                     device=self.device, dtype=torch.long)
                
                # Calculate loss using cross entropy
                loss = nn.CrossEntropyLoss()(similarity_scores, targets)
                
                # Backpropagation
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_loss = total_loss / len(dataloader)
            print(f"\nEpoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Update document embeddings after each epoch
            print("Updating document embeddings...")
            self.model.eval()
            with torch.no_grad():
                # Create progress bar for document encoding
                doc_chunks = [self.doc_texts[i:i+batch_size] 
                            for i in range(0, len(self.doc_texts), batch_size)]
                doc_pbar = tqdm(doc_chunks, desc='Encoding documents', 
                              unit='batch', position=0, leave=True)
                
                new_embeddings = []
                for doc_chunk in doc_pbar:
                    chunk_embeddings = self.encode_batch(doc_chunk)
                    new_embeddings.append(chunk_embeddings)
                
                self.doc_embeddings = np.vstack(new_embeddings)
                # Update tensor version of document embeddings
                doc_embeddings_tensor = torch.tensor(self.doc_embeddings, device=self.device)
                
                # Save document embeddings
                np.save(os.path.join(project_dir, 'models', 'dense_retriever', 'doc_embeddings.npy'), 
                       self.doc_embeddings)
                
                # Update FAISS index
                print("Updating FAISS index...")
                self.index.reset()
                self.index.add(self.doc_embeddings)
            
            # Set model back to training mode
            self.model.train()
        
        # Move model back to CPU if it was on GPU
        self.model.to('cpu')
        print("Training completed")
        
    def encode_batch(self, texts: List[str], is_training: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode a batch of texts into dense vectors
        
        Args:
            texts: List of texts to encode
            is_training: Whether this is called during training
            
        Returns:
            Array of encoded vectors (numpy array) or tensor with gradients
        """
        if is_training:
            # During training, we need to compute gradients
            features = self.model.tokenize(texts)
            features = {k: v.to(self.device) for k, v in features.items()}
            embeddings = self.model(features)['sentence_embedding']
            return embeddings
        else:
            # During inference, we can use numpy arrays
            embeddings = self.model.encode(texts, 
                                         batch_size=32,
                                         show_progress_bar=False,
                                         convert_to_numpy=True,
                                         normalize_embeddings=True)
            return embeddings
    
    def retrieve(self, 
                query: str,
                top_k: int = 5,
                **kwargs) -> Dict[str, Any]:
        """
        Retrieve most relevant documents for query
        
        Args:
            query: Query text
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing query and list of top-k document IDs
        """
        if not self.index:
            raise ValueError("Index not built. Call build_index first.")
            
        # Encode query
        query_vector = self.encode_batch([query])
        
        # Search index
        scores, doc_indices = self.index.search(query_vector, top_k)
        
        # Get document IDs for valid indices
        retrieved_ids = []
        for doc_idx in doc_indices[0]:
            if doc_idx != -1:  # FAISS may return -1 if not enough results
                retrieved_ids.append(self.doc_ids[doc_idx])
                
        # Format results according to specified format
        result = {
            "question": query,
            "answer": "",
            "document_id": retrieved_ids
        }
            
        return result
    
    def save_index(self, path: str) -> None:
        """
        Save FAISS index and document mappings to disk
        
        Args:
            path: Directory path to save to
        """
        if not self.index:
            raise ValueError("No index to save")
            
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, 'faiss.index'))
        
        # Save document mappings and embeddings
        np.save(os.path.join(path, 'doc_ids.npy'), self.doc_ids)
        np.save(os.path.join(path, 'doc_texts.npy'), self.doc_texts)
        np.save(os.path.join(path, 'doc_embeddings.npy'), self.doc_embeddings)
        
        # Save model
        self.model.save(os.path.join(path, 'model'))
        
    def load_index(self, path: str) -> None:
        """
        Load FAISS index and document mappings from disk
        
        Args:
            path: Directory path to load from
        """
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, 'faiss.index'))
        
        # Load document mappings and embeddings
        self.doc_ids = np.load(os.path.join(path, 'doc_ids.npy')).tolist()
        self.doc_texts = np.load(os.path.join(path, 'doc_texts.npy')).tolist()
        self.doc_embeddings = np.load(os.path.join(path, 'doc_embeddings.npy'))
        
        # Load model
        model_path = os.path.join(path, 'model')
        if os.path.exists(model_path):
            self.model = SentenceTransformer(model_path) 