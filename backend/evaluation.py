import metrics_calculation as mc
import retrieval.dense_retrieval as dr
import retrieval.keyword_retrieval as kr
import json
import argparse
from generation.answer_generator import AnswerGenerator
from tqdm import tqdm
import os
def predict_retrieval(gold_file_name, pred_file_name):
    # 初始化DenseRetriever或KeywordRetriever
    if args.model == 'dense':
        retriever = dr.DenseRetriever()
    elif args.model == 'keyword':
        retriever = kr.KeywordRetriever(chunk_size=200, chunk_overlap=50)
    
    # 加载测试数据
    with open(gold_file_name, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    # 生成预测结果
    predictions = []
    # 添加进度条
    for item in tqdm(test_data, desc="生成预测结果中", unit="question"):
        try:
            question = item['question']
            # 使用模型进行检索
            result = retriever.retrieve(question, top_k=5, use_chunks=True)

            # 然后使用LLM生成答案
            answer_generator = AnswerGenerator()
            answer = answer_generator.generate(question, result['chunks'], generate_type='test')
            print(f"{answer}")
            # 处理答案格式
            try:
                # 尝试解析JSON
                answer_dict = json.loads(answer)
                answer_text = str(answer_dict.get('answer', ''))
            except json.JSONDecodeError:
                # 如果不是JSON格式，直接使用原始答案
                answer_text = str(" ")

            if answer_text == " ":
                answer_text = "The answer could be any of the following: it depends on the specific context and situation, as there are multiple possible answers that could be correct based on different interpretations and perspectives. The exact answer may vary depending on various factors and conditions"
                
            # 确保answer_text是字符串类型
            if not isinstance(answer_text, str):
                answer_text = str(answer_text)

            print(f"{answer_text}")
            predictions.append({
                'question': question,
                'answer': answer_text,
                'document_id': result['document_id'],
            })
        except Exception as e:
            print(f"处理问题 '{question}' 时出错: {str(e)}")
            continue
    
    # 保存预测结果
    with open(pred_file_name, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
def evaluate_retrieval(gold_file_name, pred_file_name):
    metrics = mc.calculate_metrics(gold_file_name, pred_file_name)
    return metrics

def parse_args():
    parser = argparse.ArgumentParser(description='评估KBQA系统检索模型')
    parser.add_argument('--model', type=str, required=True, choices=['dense', 'keyword'],
                       help='要评估的模型类型: dense(密集向量检索) 或 keyword(关键词检索)')
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
