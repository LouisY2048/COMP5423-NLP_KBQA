"""
Data Loader module for the Knowledge Base Question Answering System.
This module provides functionality to load and manage document data and training/validation/test datasets.
It handles JSONL file formats and provides easy access to document content and metadata.

Key Features:
1. Loads documents from JSONL files
2. Supports training, validation, and test data loading
3. Provides document retrieval by ID
4. Handles data preprocessing and formatting

Usage:
    loader = DataLoader()
    documents = loader.get_all_documents()
    train_data = loader.get_train_data()
    val_data = loader.get_val_data()
    test_data = loader.get_test_data()
"""

import json
import os

class DataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.documents = {}
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        self._load_documents()
        
        self._load_train_data()
        self._load_val_data()
        self._load_test_data()
        
    
    def _load_documents(self):
        """Load document data"""
        doc_path = os.path.join(self.data_dir, 'documents_modified.jsonl')
        print(f"Loading document data from {doc_path}")
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                self.documents[doc['document_id']] = doc['document_text']
    
    def _load_train_data(self):
        """Load training data"""
        train_path = os.path.join(self.data_dir, 'train.jsonl')
        
        if os.path.exists(train_path):
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.train_data.append(json.loads(line))
    
    def _load_val_data(self):
        """Load validation data"""
        val_path = os.path.join(self.data_dir, 'val.jsonl')
        
        if os.path.exists(val_path):
            with open(val_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.val_data.append(json.loads(line))
    
    def _load_test_data(self):
        """Load test data"""
        test_path = os.path.join(self.data_dir, 'test.jsonl')
        
        if os.path.exists(test_path):
            with open(test_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.test_data.append(json.loads(line))
    
    def get_document_by_id(self, doc_id):
        """Get document content by ID"""
        return self.documents.get(doc_id, "")
    
    def get_all_documents(self):
        """Get all documents"""
        return self.documents
    
    def get_document_ids(self):
        """Get all document IDs"""
        return list(self.documents.keys())
    
    def get_train_data(self):
        """Get training data"""
        return self.train_data
    
    def get_val_data(self):
        """Get validation data"""
        return self.val_data
    
    def get_test_data(self):
        """Get test data"""
        return self.test_data 