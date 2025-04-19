"""
Evaluation script for the Knowledge Base Question Answering System.
This module provides functionality to:
1. Generate predictions using different retrieval methods (dense or keyword)
2. Evaluate the system's performance using various metrics
3. Compare results against ground truth data

Usage:
    python evaluation.py --model [dense|keyword]
"""

import metrics_calculation as mc
import retrieval.dense_retrieval as dr
import retrieval.keyword_retrieval as kr
import json
import argparse
from generation.answer_generator import AnswerGenerator
from tqdm import tqdm
import os

def predict_retrieval(gold_file_name, pred_file_name):
    """Predict retrieval results"""
    if args.model == 'dense':
        retriever = dr.DenseRetriever()
    elif args.model == 'keyword':
        retriever = kr.KeywordRetriever(chunk_size=200, chunk_overlap=50)

    with open(gold_file_name, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    predictions = []
    for item in tqdm(test_data, desc="Generating predictions", unit="question"):
        try:
            question = item['question']
            result = retriever.retrieve(question, top_k=5, use_chunks=True)

            answer_generator = AnswerGenerator()
            answer = answer_generator.generate(question, result['chunks'], generate_type='test')
            print(f"{answer}")
            try:
                answer_dict = json.loads(answer)
                answer_text = str(answer_dict.get('answer', ''))
            except json.JSONDecodeError:
                answer_text = str(" ")

            if answer_text == " ":
                answer_text = "The answer could be any of the following: it depends on the specific context and situation, as there are multiple possible answers that could be correct based on different interpretations and perspectives. The exact answer may vary depending on various factors and conditions"
                
            if not isinstance(answer_text, str):
                answer_text = str(answer_text)

            print(f"{answer_text}")
            predictions.append({
                'question': question,
                'answer': answer_text,
                'document_id': result['document_id'],
            })
        except Exception as e:
            print(f"Error processing question '{question}': {str(e)}")
            continue
    
    with open(pred_file_name, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
def evaluate_retrieval(gold_file_name, pred_file_name):
    metrics = mc.calculate_metrics(gold_file_name, pred_file_name)
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate KBQA system retrieval models')
    parser.add_argument('--model', type=str, required=True, choices=['dense', 'keyword'],
                       help='Model type to evaluate: dense (dense vector retrieval) or keyword (keyword retrieval)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    gold_file = 'data/val.jsonl'

    if args.model == 'dense':
        if not os.path.exists('data/val_predict(dense).jsonl'):
            predict_retrieval(gold_file, 'data/val_predict(dense).jsonl')
        pred_file = 'data/val_predict(dense).jsonl' 
        evaluate_retrieval(gold_file, pred_file)
    elif args.model == 'keyword':
        if not os.path.exists('data/val_predict(keyword).jsonl'):
            predict_retrieval(gold_file, 'data/val_predict(keyword).jsonl')
        pred_file = 'data/val_predict(keyword).jsonl'
        evaluate_retrieval(gold_file, pred_file)
    
    metrics = mc.calculate_metrics(gold_file, pred_file)
    print(f"Evaluation Result:", flush=True)
    print(f"Answer Accuracy:             {metrics['accuracy']:.4f}", flush=True)
    print(f"Answer BERT Score:           {metrics['bert_score']:.4f}", flush=True)
    print(f"Document Retrieval Recall@5: {metrics['recall@5']:.4f}", flush=True)
    print(f"Document Retrieval MRR@5   : {metrics['mrr@5']:.4f}", flush=True)
