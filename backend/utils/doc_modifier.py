"""
Document Modifier module for the Knowledge Base Question Answering System.
This module provides functionality to preprocess and modify document content.
It handles text cleaning, chunking, and formatting for better document processing.

Key Features:
1. Cleans HTML and special characters from text
2. Splits documents into manageable chunks
3. Removes unwanted sections (e.g., "Contents", "See also")
4. Supports batch processing with GPU acceleration
5. Handles document formatting and structure
"""

import os
import json
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from nltk import sent_tokenize

class DocumentModifier:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        self.whitespace_re = re.compile(r'\s+')
        self.control_chars_re = re.compile(r'[\x00-\x1F\x7F-\x9F]')
        self.url_re = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_re = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.punct_re = re.compile(r'[^\w\s.,!?-]')
        self.repeat_punct_re = re.compile(r'([.,!?])\1+')
        self.stopwords_re = re.compile(r'\b(the|and|is|in|to|of|a|that|it|with|as|for|by|on|at|from|up|down|out|about|into|over|under|again|further|then|once|here|there|when|where|why|how|all|any|some|one|two|three|four|five|six|seven|eight|nine|ten)\b')
    
    def clean_text(self, text):
        """Clean up HTML tags and other formatting in text"""
        soup = BeautifulSoup(text, 'html.parser')
        contents_h2 = soup.find('h2', string=lambda text: text and 'Contents' in text)
        if contents_h2:
            next_h2 = contents_h2.find_next('h2')
            if next_h2:
                current = contents_h2
                while current and current != next_h2:
                    next_sibling = current.next_sibling
                    current.decompose()
                    current = next_sibling
            else:
                current = contents_h2
                while current:
                    next_sibling = current.next_sibling
                    current.decompose()
                    current = next_sibling
        delete_h2_list = ['See to', 'See also']
        
        for h2 in soup.find_all('h2'):
            h2_text = h2.get_text()
            if any(keyword in h2_text for keyword in delete_h2_list):
                current = h2
                while current:
                    next_sibling = current.next_sibling
                    current.decompose()
                    current = next_sibling
        
        text = soup.get_text()
        
        text = re.sub(r'\s+', ' ', text)
        
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        
        text = text.strip()
        
        return text
    
    def clean_batch(self, texts, batch_size=32):
        """Batch clean text"""
        if self.device == 'cuda':
            results = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_results = list(executor.map(self.clean_text, batch))
                    results.extend(batch_results)
            return results
        else:
            return [self.clean_text(text) for text in texts]
    
    def split_documents(self, input_file=None, output_file=None, chunk_size=500, chunk_overlap=0):
        """Split documents and save"""
        if input_file is None:
            input_file = os.path.join(self.project_dir, 'data', 'documents_modified.jsonl')
        
        if output_file is None:
            output_dir = os.path.dirname(input_file)
            output_file = os.path.join(output_dir, 'documents_splitted.jsonl')
        
        print(f"Start splitting documents...")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        chunk_id = 0
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, desc="Processing progress"):
                doc = json.loads(line)
                doc_id = doc['document_id']
                text = doc['document_text']
                
                sentences = sent_tokenize(text)
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    
                    if current_length + sentence_length > chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunk_data = {
                            "chunk_id": chunk_id,
                            "document_id": doc_id,
                            "chunk_text": chunk_text
                        }
                        f_out.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
                        chunk_id += 1
                        
                        current_chunk = []
                        current_length = 0
                    
                    current_chunk.append(sentence)
                    current_length += sentence_length
                
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "chunk_text": chunk_text
                    }
                    f_out.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
                    chunk_id += 1
        
        print(f"Document splitting completed!")
        print(f"Total {chunk_id} text blocks generated")
        print(f"Splitting results saved to: {output_file}")

    def process_documents(self, input_file=None, output_file=None):
        """Process document file"""
        if input_file is None:
            input_file = os.path.join(self.project_dir, 'data', 'documents.jsonl')
        
        if output_file is None:
            output_dir = os.path.dirname(input_file)
            output_file = os.path.join(output_dir, 'documents_modified.jsonl')
        
        print(f"Start processing documents...")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        modified_docs = []
        total_docs = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        removed_content_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_docs, desc="Processing progress"):
                doc = json.loads(line)
                
                if 'document_text' in doc:
                    original_length = len(doc['document_text'])
                    doc['document_text'] = self.clean_text(doc['document_text'])
                    if len(doc['document_text']) < original_length:
                        removed_content_count += 1
                
                modified_docs.append(doc)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in modified_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"Processing completed! Processed {len(modified_docs)} documents")
        print(f"Processed documents saved to: {output_file}")
    
def main():
    modifier = DocumentModifier()
    modifier.process_documents()
    print("\nStart splitting documents...")
    modifier.split_documents()

if __name__ == "__main__":
    main() 