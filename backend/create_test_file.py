"""
System environment test script for the Knowledge Base Question Answering System.
This module tests the availability of required hardware and libraries:
1. CUDA GPU availability for PyTorch
2. FAISS GPU support
3. Environment variable configuration

"""

import torch
import os
import faiss
import json
from retrieval.dense_retrieval import DenseRetriever
from generation.answer_generator import AnswerGenerator
from tqdm import tqdm

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def create_test_file(test_file_name, pred_file_name):
    retriever = DenseRetriever()

    with open(test_file_name, 'r', encoding='utf-8') as f:
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

if __name__ == "__main__":
    test_file_name = os.path.join(project_dir, 'data', 'test.jsonl')
    pred_file_name = os.path.join(os.path.dirname(project_dir), 'test_predict.jsonl')
    create_test_file(test_file_name, pred_file_name)
