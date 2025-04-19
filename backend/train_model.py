"""
Model training script for the Knowledge Base Question Answering System.
This module provides functionality to train and save retrieval models:
1. Dense Retrieval model (semantic retrieval based on Sentence Transformers)
2. Keyword Retrieval model (keyword-based retrieval using TF-IDF)

Usage:
    python train_model.py --model [dense|keyword] [options]

Options:
    --model: Model type to train: dense (dense vector retrieval) or keyword (keyword retrieval)
    --train_file: Path to training data file
    --docs_file: Path to document data file
    --output_dir: Directory to save trained models
    --device: Training device (cuda/cpu)
    --max_features_range: Range of TF-IDF max_features parameters
    --epochs: Number of training epochs
    --chunk_size: Chunk size for keyword retrieval
    --chunk_overlap: Chunk overlap for keyword retrieval
    --top_k: Top k for retrieval
"""

import os
import sys
import json
import argparse
from datetime import datetime
from retrieval.dense_retrieval import DenseRetriever
from retrieval.keyword_retrieval import KeywordRetriever
import torch
from typing import List, Dict, Any

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

"""
python train_model.py --model dense
python train_model.py --model keyword
"""
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from jsonl file"""
    if not os.path.isabs(file_path):
        file_path = os.path.join(project_dir, file_path)
    
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
            'document_text': item['document_text']
        }
        documents.append(doc)
    return documents

def load_training_data(file_path):
    """Load training data"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(sample)
        print(f"Successfully loaded {len(data)} training data")
        return data
    except Exception as e:
        print(f"Error loading training data: {e}")
        sys.exit(1)

def train_dense_retriever(args):
    """Train Dense Retrieval model"""
    print("\n" + "="*50)
    print(f"Start training Dense Retrieval model")
    print("="*50)

    print("Loading documents and training data...")
    print(f"Loading cleaned documents from {args.docs_file}...")
    documents = load_jsonl(args.docs_file)
    print(f"Loading training data from {args.train_file}...")
    train_data = load_jsonl(args.train_file)
    
    print(f"Loaded {len(documents)} cleaned documents and {len(train_data)} training data")
    
    chunks_file = os.path.join(project_dir, 'data', 'documents_splitted.jsonl')
    if not os.path.exists(chunks_file):
        print(f"Warning: Pre-chunked file not found: {chunks_file}")
        print("Please run the split_documents method in doc_modifier.py to generate document chunks")
        sys.exit(1)
    else:
        print(f"Found pre-chunked file: {chunks_file}")
        chunk_count = sum(1 for _ in open(chunks_file, 'r', encoding='utf-8'))
        print(f"Contains {chunk_count} text blocks")
    
    print("Initializing DenseRetriever...")
    retriever = DenseRetriever(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        index_type='Flat',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print("Preparing documents...")
    doc_data = prepare_documents(documents)
    print(f"Prepared documents, document count: {len(doc_data)}")
    
    print("Building index and training...")
    try:
        retriever.build_index(documents=doc_data, train_data=train_data)
    except Exception as e:
        print(f"Error building index or training: {e}")
        sys.exit(1)
    
    output_dir = os.path.join(project_dir, args.output_dir.lstrip('/'))
    print(f"Saving model and index to {output_dir}")
    try:
        retriever.save_index(output_dir)
        print("Completed!")
    except Exception as e:
        print(f"Error saving model and index: {e}")
        sys.exit(1)

    output_path = os.path.join(output_dir, "dense_retriever")
    
    config_path = os.path.join(output_path, 'dense_training_config.json')
    config = {
        'model_type': 'dense_retrieval',
        'train_file': os.path.abspath(args.train_file),
        'docs_file': os.path.abspath(args.docs_file),
        'chunks_file': chunks_file,
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'index_type': 'Flat',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'train_samples': len(train_data),
        'doc_count': len(documents),
        'chunk_count': chunk_count,
        'training_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'document_flow': {
            'original_docs': os.path.join(project_dir, 'data', 'documents.jsonl'),
            'cleaned_docs': os.path.join(project_dir, 'data', 'documents_modified.jsonl'),
            'split_docs': chunks_file
        }
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"\nTraining completed!")
    print(f"Model and index saved to: {output_path}")
    print(f"Training config saved to: {config_path}")
    return output_path

def train_keyword_retriever(args):
    """Train Keyword Retrieval model (TF-IDF parameter optimization)"""
    print("\n" + "="*50)
    print(f"Start training Keyword Retrieval model (TF-IDF parameter optimization)")
    print("="*50)
    
    train_data = load_training_data(args.train_file)
    
    output_path = os.path.join(args.output_dir, "keyword_retriever")
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, 'tfidf_model.pkl')
    
    print(f"Initializing Keyword Retriever (chunk_size: {args.chunk_size})")
    retriever = KeywordRetriever(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k
    )
    
    print("\nTraining model (parameter optimization)...")
    best_params = retriever.train(
        training_data=train_data,
        epochs=args.epochs
    )
    
    retriever.save_model(model_path)
    print(f"\nTraining completed! Model saved to: {model_path}")
    
    config_path = os.path.join(output_path, 'tfidf_training_config.json')
    config = {
        'model_type': 'keyword_tfidf',
        'train_file': args.train_file,
        'best_params': best_params,
        'max_features_range': args.max_features_range,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'top_k': args.top_k,
        'train_samples': len(train_data),
        'model_path': model_path
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Training config saved to: {config_path}")
    return output_path

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train KBQA system retrieval models')
    
    parser.add_argument('--model', type=str, required=True, choices=['dense', 'keyword'],
                       help='要训练的模型类型: dense(密集向量检索) 或 keyword(关键词检索)')
    parser.add_argument('--train_file', type=str, default='data/train.jsonl',
                       help='训练数据文件路径')
    parser.add_argument('--docs_file', type=str, default='data/documents_modified.jsonl',
                       help='文档数据文件路径')
    parser.add_argument('--output_dir', type=str, default='/backend/models',
                       help='模型输出目录')
    parser.add_argument('--top_k', type=int, default=5,
                       help='检索时返回的文档数量')
    
    parser.add_argument('--chunk_size', type=int, default=50,
                       help='文本分块大小(单词数)')
    parser.add_argument('--chunk_overlap', type=int, default=50,
                       help='文本分块重叠大小(单词数)')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备 (cuda/cpu)')
    
    parser.add_argument('--max_features_range', type=int, nargs='+', default=[5000, 10000, 15000, 20000],
                       help='TF-IDF max_features parameter range')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    if args.model == 'dense':
        output_path = train_dense_retriever(args)
    elif args.model == 'keyword':
        output_path = train_keyword_retriever(args)
    
    print("\n" + "="*50)
    print(f"Model training and saving completed!")
    print(f"Model path: {output_path}")
    print("="*50)

if __name__ == "__main__":
    main() 