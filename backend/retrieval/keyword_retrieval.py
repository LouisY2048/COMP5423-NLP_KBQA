"""
Keyword Retrieval module for the Knowledge Base Question Answering System.
This module implements keyword-based document retrieval using TF-IDF vectorization.
It provides efficient text-based search capabilities by analyzing term frequency and inverse document frequency.

Key Features:
1. Uses TF-IDF for document and query vectorization
2. Implements efficient text similarity search
3. Supports document chunking and overlap
4. Provides model saving and loading functionality
5. GPU acceleration support for faster processing

Usage:
    retriever = KeywordRetriever()
    result = retriever.retrieve(query,top_k=5,use_chunks=(True or False))
"""

import os
import json
import re
import nltk
import numpy as np
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.utils.data_loader import DataLoader
from backend.generation.answer_generator import AnswerGenerator
from backend.retrieval.base_retriever import BaseRetriever
from tqdm import tqdm
import scipy.sparse as sp
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class KeywordRetriever(BaseRetriever):
    def __init__(self, chunk_size, chunk_overlap, top_k=5, load_from='backend/models/keyword_retriever'):
        super().__init__(top_k, chunk_size, chunk_overlap)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = 'english'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.data_loader = DataLoader()
        self.doc_ids = []
        self.doc_texts = []
        self.doc_processed = []
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        self.tfidf_matrix = None

        load_from = os.path.join(project_dir, 'backend', 'models', 'keyword_retriever')

        if load_from and os.path.exists(load_from):
            print(f"Loading TF-IDF retriever from Model Directory...")
            self.load_model(load_from)
        else:
            print("Initializing TF-IDF retriever...")
            self._initialize_tfidf()
            print("TF-IDF retriever initialized")
    
    def _preprocess_text(self, text):
        """Preprocess text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def _preprocess_batch(self, texts):
        """Batch preprocess text"""
        if self.device == 'cuda':
            processed_texts = []
            batch_size = 32  # 可以根据GPU内存调整
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                with ThreadPoolExecutor() as executor:
                    processed_batch = list(executor.map(self._preprocess_text, batch))
                processed_texts.extend(processed_batch)
        else:
            processed_texts = [self._preprocess_text(text) for text in texts]
        
        return processed_texts
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF model"""
        documents = self.data_loader.get_all_documents()

        self.doc_ids = list(documents.keys())
        self.doc_texts = list(documents.values())
        
        print("Preprocess documents...")
        self.doc_processed = self._preprocess_batch(self.doc_texts)
        
        print("Create TF-IDF model...")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            token_pattern=r'(?u)\b\w+\b',
            max_features=10000
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.doc_processed)
        print("TF-IDF model initialized")
    
    def retrieve(self, 
                query: str,
                top_k: int = 5,
                use_chunks: bool = False,
                **kwargs)-> Dict[str, Any]:
        """Retrieve relevant documents based on the question"""
        processed_question = self._preprocess_text(query)
        
        scores = self._get_doc_scores(processed_question)
        
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        retrieved_doc_ids = [doc_id for doc_id, _ in sorted_docs[:top_k]]

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
        """Answer question based on the retrieved document IDs"""
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
        for chunks in relevant_chunks.values():
            all_chunks.extend(chunks)
        
        result = {
            "question": query,
            "answer": "",
            "document_id": retrieved_doc_ids,
            "chunks": all_chunks
        }
        
        return result
    
    def _get_doc_scores(self, processed_question):
        """Get the relevance score of each document to the question"""
        question_vector = self.tfidf_vectorizer.transform([processed_question])
        
        if self.device == 'cuda':
            question_tensor = torch.from_numpy(question_vector.toarray()).float().to(self.device)
            doc_tensor = torch.from_numpy(self.tfidf_matrix.toarray()).float().to(self.device)
            
            question_norm = torch.norm(question_tensor, dim=1, keepdim=True)
            doc_norm = torch.norm(doc_tensor, dim=1, keepdim=True)
            
            scores = torch.mm(question_tensor, doc_tensor.t())
            
            scores = scores / (question_norm * doc_norm.t())
            
            scores = scores.cpu().numpy()[0]
        else:
            scores = (question_vector @ self.tfidf_matrix.T).toarray()[0]
        
        return {doc_id: float(score) for doc_id, score in zip(self.doc_ids, scores)}

    def train(self, training_data=None, epochs=5):
        """Optimize TF-IDF parameters"""
        if training_data is None:
            print("No training data provided, using default parameters")
            return {}
        
        print("Start optimizing TF-IDF parameters...")
        best_recall = 0
        best_params = {}
        
        print("Preprocess all documents...")
        processed_docs = self.doc_processed
        
        max_features_range = [5000, 10000, 15000, 20000]
        
        for max_features in tqdm(max_features_range, desc="Parameter optimization"):
            test_vectorizer = TfidfVectorizer(
                stop_words=self.stop_words,
                token_pattern=r'(?u)\b\w+\b',
                max_features=max_features
            )
            
            test_matrix = test_vectorizer.fit_transform(processed_docs)
            
            if self.device == 'cuda':
                doc_tensor = torch.from_numpy(test_matrix.toarray()).float().to(self.device)
                
                recall_sum = 0
                batch_size = 32
                
                for i in range(0, len(training_data), batch_size):
                    batch = training_data[i:min(i + batch_size, len(training_data))]
                    batch_questions = [sample['question'] for sample in batch]
                    batch_gold_ids = [sample['document_id'] for sample in batch]
                    
                    processed_questions = [self._preprocess_text(q) for q in batch_questions]
                    question_vectors = test_vectorizer.transform(processed_questions)
                    question_tensor = torch.from_numpy(question_vectors.toarray()).float().to(self.device)
                    
                    question_norm = torch.norm(question_tensor, dim=1, keepdim=True)
                    doc_norm = torch.norm(doc_tensor, dim=1, keepdim=True)
                    scores = torch.mm(question_tensor, doc_tensor.t())
                    scores = scores / (question_norm * doc_norm.t())
                    
                    top_k_scores, top_k_indices = torch.topk(scores, k=self.top_k, dim=1)
                    top_k_indices = top_k_indices.cpu().numpy()
                    
                    for idx, (gold_ids, retrieved_indices) in enumerate(zip(batch_gold_ids, top_k_indices)):
                        if not isinstance(gold_ids, list):
                            gold_ids = [gold_ids]
                        retrieved_doc_ids = [self.doc_ids[i] for i in retrieved_indices]
                        hits = sum(1 for doc_id in gold_ids if doc_id in retrieved_doc_ids)
                        recall = hits / len(gold_ids) if gold_ids else 0
                        recall_sum += recall
            else:
                recall_sum = 0
                for sample in tqdm(training_data, desc=f"Evaluate max_features={max_features}", leave=False):
                    question = sample['question']
                    gold_doc_ids = sample['document_id']
                    
                    if not isinstance(gold_doc_ids, list):
                        gold_doc_ids = [gold_doc_ids]
                    
                    processed_question = self._preprocess_text(question)
                    question_vector = test_vectorizer.transform([processed_question])
                    doc_scores = (question_vector @ test_matrix.T).toarray()[0]
                    top_indices = np.argsort(doc_scores)[::-1][:self.top_k]
                    retrieved_doc_ids = [self.doc_ids[idx] for idx in top_indices]
                    
                    hits = sum(1 for doc_id in gold_doc_ids if doc_id in retrieved_doc_ids)
                    recall = hits / len(gold_doc_ids) if gold_doc_ids else 0
                    recall_sum += recall
            
            avg_recall = recall_sum / len(training_data)
            print(f"max_features={max_features}  Recall@{self.top_k}: {avg_recall:.4f}")
            
            if avg_recall > best_recall:
                best_recall = avg_recall
                best_params = {'max_features': max_features}
                print(f"✓ Found better parameters: {best_params}, Recall: {best_recall:.4f}")
        
        print(f"\nTraining completed! Best parameters: {best_params}, Best recall: {best_recall:.4f}")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=self.stop_words,
            token_pattern=r'(?u)\b\w+\b',
            max_features=best_params.get('max_features', 10000)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_docs)
        
        self.save_model()
        
        return best_params
    
    def save_model(self, path=None):
        """Save model to file"""
        if path is None:
            path = os.path.join(project_dir, 'backend', 'models', 'keyword_retriever')
        
        os.makedirs(path, exist_ok=True)
        
        config_data = {
            'stop_words': self.stop_words,
            'top_k': int(self.top_k),
            'chunk_size': int(self.chunk_size),
            'chunk_overlap': int(self.chunk_overlap),
            'tfidf_params': {
                'stop_words': self.tfidf_vectorizer.stop_words,
                'max_features': int(self.tfidf_vectorizer.max_features),
                'token_pattern': self.tfidf_vectorizer.token_pattern,
            }
        }
        with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        doc_data = {
            'doc_ids': [int(id_) if isinstance(id_, (np.integer, np.floating)) else id_ for id_ in self.doc_ids],
            'doc_texts': self.doc_texts
        }
        with open(os.path.join(path, 'documents.json'), 'w', encoding='utf-8') as f:
            json.dump(doc_data, f, ensure_ascii=False, indent=2)
        
        vectorizer_data = {
            'vocabulary': {k: int(v) for k, v in self.tfidf_vectorizer.vocabulary_.items()},
            'idf': [float(x) for x in self.tfidf_vectorizer.idf_],
            'feature_names': [str(x) for x in self.tfidf_vectorizer.get_feature_names_out()]
        }
        with open(os.path.join(path, 'vectorizer.json'), 'w', encoding='utf-8') as f:
            json.dump(vectorizer_data, f, ensure_ascii=False, indent=2)
        
        if isinstance(self.tfidf_matrix, sp.csr_matrix):
            matrix_data = self.tfidf_matrix.toarray().tolist()
        else:
            matrix_data = [[float(x) for x in row] for row in self.tfidf_matrix.tolist()]
        with open(os.path.join(path, 'matrix.json'), 'w', encoding='utf-8') as f:
            json.dump(matrix_data, f, ensure_ascii=False)
        
        print(f"TF-IDF model and document blocks saved to {path}")
    
    def load_model(self, path=None):
        """Load model from file"""
        if path is None:
            path = os.path.join(project_dir, 'backend', 'models', 'keyword_retriever')
        
        with open(os.path.join(path, 'config.json'), 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        self.stop_words = config_data['stop_words']
        self.top_k = config_data['top_k']
        self.chunk_size = config_data['chunk_size']
        self.chunk_overlap = config_data['chunk_overlap']
        
        with open(os.path.join(path, 'documents.json'), 'r', encoding='utf-8') as f:
            doc_data = json.load(f)
        self.doc_ids = doc_data['doc_ids']
        self.doc_texts = doc_data['doc_texts']
        
        with open(os.path.join(path, 'vectorizer.json'), 'r', encoding='utf-8') as f:
            vectorizer_data = json.load(f)
        
        self.tfidf_vectorizer = TfidfVectorizer(**config_data['tfidf_params'])
        self.tfidf_vectorizer.vocabulary_ = vectorizer_data['vocabulary']
        self.tfidf_vectorizer.idf_ = np.array(vectorizer_data['idf'])
        self.tfidf_vectorizer._tfidf._idf_diag = sp.diags(
            self.tfidf_vectorizer.idf_,
            offsets=0,
            shape=(len(self.tfidf_vectorizer.idf_), len(self.tfidf_vectorizer.idf_)),
            format='csr',
            dtype=np.float64
        )
        
        with open(os.path.join(path, 'matrix.json'), 'r', encoding='utf-8') as f:
            matrix_data = json.load(f)
        self.tfidf_matrix = sp.csr_matrix(matrix_data)
        
        print(f"TF-IDF model and document blocks loaded from {path}") 

if __name__ == "__main__":
    print("project_dir: ", project_dir)

