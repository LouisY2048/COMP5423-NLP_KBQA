"""
Dense Retrieval module for the Knowledge Base Question Answering System.
This module implements dense vector-based document retrieval using sentence transformers.
It provides efficient semantic search capabilities by encoding documents and queries into dense vectors.

Key Features:
1. Uses sentence-transformers for document and query encoding
2. Implements efficient vector similarity search
3. Supports document chunking and overlap
4. Provides model saving and loading functionality

Usage:
    retriever = DenseRetriever()
    result = retriever.retrieve(query,top_k=5,use_chunks=(True or False))
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
import json

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainDataset(Dataset):
    """Dataset for training dense retriever"""
    def __init__(self, questions: List[str], targets: List[Any], texts: Optional[List[str]] = None):
        """
        Initialize dataset
        
        Args:
            questions: List of questions
            targets: List of targets (document IDs for doc-level, labels for chunk-level)
            texts: List of texts (None for doc-level, chunk texts for chunk-level)
        """
        self.questions = questions
        self.targets = targets
        self.texts = texts
        
    def __len__(self):
        return len(self.questions)
        
    def __getitem__(self, idx):
        if self.texts is None:
            # Document-level training
            return self.questions[idx], self.targets[idx]
        else:
            # Chunk-level training
            return self.questions[idx], self.texts[idx], self.targets[idx]

class DenseRetriever(BaseRetriever):
    """Dense retriever using FAISS for efficient similarity search"""
    
    def __init__(self, 
                 model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 index_type: str = 'Flat',
                 device: str = 'cuda',
                 **kwargs):
        """
        Initialize the dense retriever
        
        Args:
            model_name: Name/path of the sentence transformer model to use
            index_type: Type of FAISS index to use ('Flat' or 'IVF')
            device: Device to use for encoding ('cpu' or 'cuda')
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between adjacent chunks
            **kwargs: Additional arguments
        """
        super().__init__()
        self.model_name = model_name
        self.index_type = index_type
        self.device = device
        
        self.model = SentenceTransformer(model_name)
        if device == 'cuda':
            self.model = self.model.to('cuda')
            
        self.index = None
        self.doc_ids = []
        self.doc_texts = []
        self.doc_embeddings = None

        if os.path.exists(os.path.join(project_dir, 'backend', 'models', 'dense_retriever')):
            self.load_index()
        else:
            print("Please run the build_index method and provide document data to create an index")
        
    def build_index(self, documents: List[Dict[str, Any]], train_data: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Build FAISS index from documents and optionally train with training data
        
        Args:
            documents: List of document dictionaries with 'document_id' and 'document_text' fields
            train_data: Optional list of training examples with 'question' and 'document_id' fields
        """
        if os.path.exists(os.path.join(project_dir, 'backend', 'models', 'dense_retriever')):
            self.load_index()
        else:
            print("Building index...")
            
            print("Extracting document texts and IDs...")
            self.doc_texts = [doc['document_text'] for doc in documents]
            self.doc_ids = [doc['document_id'] for doc in documents]
            
            print("Calculating document embeddings...")
            self.doc_embeddings = self.encode_batch(self.doc_texts)
            
            print("Initializing FAISS index...")
            vector_dimension = self.doc_embeddings.shape[1]
            if self.index_type == 'Flat':
                self.index = faiss.IndexFlatIP(vector_dimension)
                print("Using Flat index (exact search)")
            elif self.index_type == 'IVF':
                nlist = min(len(documents) // 10, 100)  # 聚类中心数量
                quantizer = faiss.IndexFlatIP(vector_dimension)
                self.index = faiss.IndexIVFFlat(quantizer, vector_dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                print(f"Using IVF index (approximate search), number of clusters: {nlist}")
                print("Training FAISS index...")
                self.index.train(self.doc_embeddings)
            
            print("Adding document embeddings to FAISS index...")
            self.index.add(self.doc_embeddings)
            print(f"FAISS index now contains {self.index.ntotal} vectors")
            
            print("Saving all data (including FAISS index)...")
            self.save_index()
        
        if train_data:
            print("Training with training data...")
            self.train(train_data)
        
    
    def train(self, train_data: List[Dict[str, Any]], 
              batch_size: int = 32,
              epochs: int = 3,
              learning_rate: float = 2e-5,
              gradient_accumulation_steps: int = 2) -> None:
        """Train the dense retriever model using training data"""
        print("Starting training...")
        self.model.to(self.device)
        self.model.train()

        if isinstance(self.doc_embeddings, np.ndarray):
            self.doc_embeddings = torch.from_numpy(self.doc_embeddings).to(self.device)
        else:
            self.doc_embeddings = self.doc_embeddings.to(self.device)
    
        questions = [item['question'] for item in train_data]
        doc_ids = [item['document_id'] for item in train_data]
        
        doc_dataset = TrainDataset(questions, doc_ids)
        doc_dataloader = DataLoader(doc_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            total_doc_loss = 0
            
            doc_pbar = tqdm(doc_dataloader, desc=f'Epoch {epoch+1}/{epochs} (Doc)', 
                          unit='batch', position=0, leave=True)
            
            for batch_idx, (batch_questions, batch_doc_ids) in enumerate(doc_pbar):
                optimizer.zero_grad()
                
                question_embeddings = self.encode_batch(batch_questions, is_training=True)
                
                similarity_scores = torch.matmul(question_embeddings, self.doc_embeddings.T)
                
                targets = torch.tensor([self.doc_ids.index(doc_id) for doc_id in batch_doc_ids],
                                     device=self.device, dtype=torch.long)
                
                doc_loss = nn.CrossEntropyLoss()(similarity_scores, targets)
                
                doc_loss = doc_loss / gradient_accumulation_steps
                
                doc_loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_doc_loss += doc_loss.item() * gradient_accumulation_steps
                doc_pbar.set_postfix({'doc_loss': f'{doc_loss.item() * gradient_accumulation_steps:.4f}'})
                
                del question_embeddings, similarity_scores, targets, doc_loss
                torch.cuda.empty_cache()
            
            avg_doc_loss = total_doc_loss / len(doc_dataloader)
            print(f"\nEpoch {epoch+1}/{epochs}, Average Doc Loss: {avg_doc_loss:.4f}")
            
            print("Updating document embeddings and indices...")
            self.model.eval()
            with torch.no_grad():
                self.doc_embeddings = self.encode_batch(self.doc_texts)
                if isinstance(self.doc_embeddings, np.ndarray):
                    self.doc_embeddings = torch.from_numpy(self.doc_embeddings).to(self.device)
                else:
                    self.doc_embeddings = self.doc_embeddings.to(self.device)
                
                self.index.reset()
                if torch.is_tensor(self.doc_embeddings):
                    doc_embeddings_cpu = self.doc_embeddings.cpu()
                    self.index.add(doc_embeddings_cpu.numpy())
                else:
                    self.index.add(self.doc_embeddings)
                
                self.save_index()
            
            self.model.train()
        
        print("Training completed")
    
    def encode_batch(self, texts: List[str], is_training: bool = False) -> Union[np.ndarray, torch.Tensor]:
        """Encode a batch of texts into dense vectors"""
        if is_training:
            features = self.model.tokenize(texts)
            features = {k: v.to(self.device) for k, v in features.items()}
            embeddings = self.model(features)['sentence_embedding']
            return embeddings
        else:
            embeddings = self.model.encode(texts, 
                                         batch_size=32,
                                         show_progress_bar=False,
                                         convert_to_numpy=True,
                                         normalize_embeddings=True)
            return embeddings
    
    def retrieve(self, 
                query: str,
                top_k: int = 5,
                use_chunks: bool = False,
                **kwargs) -> Dict[str, Any]:
        """Retrieve most relevant documents for query"""
        if not self.index:
            raise ValueError("Index not built. Call build_index first.")
            
        query_vector = self.encode_batch([query])
        scores, doc_indices = self.index.search(query_vector, top_k)
        
        retrieved_doc_ids = []
        for doc_idx in doc_indices[0]:
            if doc_idx != -1:  # FAISS may return -1 if not enough results
                retrieved_doc_ids.append(self.doc_ids[doc_idx])
        
        if use_chunks:
            return self.answer_question_by_chunks(query, top_k, retrieved_doc_ids)
        
        result = {
            "question": query,
            "answer": "",
            "document_id": retrieved_doc_ids
        }
            
        return result

    
    def answer_question_by_chunks(self, 
                                  query: str,
                                  top_k: int = 5,
                                  retrieved_doc_ids: List[str] = None) -> Dict[str, Any]:
        """Answer question by retrieving relevant chunks"""
        if not retrieved_doc_ids:
            return {
                "question": query,
                "answer": "",
                "document_id": "",
                "chunks": []
            }

        chunks_file = os.path.join(project_dir, 'data', 'documents_splitted.jsonl')
        relevant_chunks = {doc_id: [] for doc_id in retrieved_doc_ids}
        
        if os.path.exists(chunks_file):
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk = json.loads(line)
                    doc_id = chunk['document_id']
                    if doc_id in relevant_chunks and len(relevant_chunks[doc_id]) < top_k:
                        relevant_chunks[doc_id].append(chunk['chunk_text'])
                    
                    if all(len(chunks) >= top_k for chunks in relevant_chunks.values()):
                        break
        
        all_chunks = []
        for doc_id in retrieved_doc_ids:
            all_chunks.extend(relevant_chunks[doc_id][:top_k])
        
        result = {
            "question": query,
            "answer": "",
            "document_id": retrieved_doc_ids,
            "chunks": all_chunks
        }
        
        return result
    
    def save_index(self, path: str = None) -> None:
        """Save FAISS index and document mappings to disk"""
        if not self.index:
            raise ValueError("No index to save")
            
        if path is None:
            path = os.path.join(project_dir, 'backend', 'models', 'dense_retriever')
        
        os.makedirs(path, exist_ok=True)
        
        doc_texts_array = np.array(self.doc_texts, dtype=object)
        doc_ids_array = np.array(self.doc_ids, dtype=object)
        
        np.save(os.path.join(path, 'doc_texts.npy'), doc_texts_array)
        np.save(os.path.join(path, 'doc_ids.npy'), doc_ids_array)
        
        if torch.is_tensor(self.doc_embeddings):
            doc_embeddings_np = self.doc_embeddings.cpu().numpy()
        else:
            doc_embeddings_np = self.doc_embeddings
            
        if doc_embeddings_np.size == 0:
            raise ValueError("doc_embeddings为空，无法保存")
            
        np.save(os.path.join(path, 'doc_embeddings.npy'), doc_embeddings_np)

        faiss.write_index(self.index, os.path.join(path, 'faiss.index'))

        print("Index saved successfully")
        
    def load_index(self, path: str = None) -> None:
        """Load FAISS index and document mappings from disk"""
        if path is None:
            path = os.path.join(project_dir, 'backend', 'models', 'dense_retriever')
        
        doc_texts_array = np.load(os.path.join(path, 'doc_texts.npy'), allow_pickle=True)
        doc_ids_array = np.load(os.path.join(path, 'doc_ids.npy'), allow_pickle=True)
        
        self.doc_texts = doc_texts_array.tolist()
        self.doc_ids = doc_ids_array.tolist()
        
        self.doc_embeddings = np.load(os.path.join(path, 'doc_embeddings.npy'))
        
        self.index = faiss.read_index(os.path.join(path, 'faiss.index'))

        print("Index loaded successfully")