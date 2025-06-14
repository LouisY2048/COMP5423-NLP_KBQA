import json
from bert_score import score

def calculate_metrics(gold_file, pred_file):
    with open(gold_file, 'r', encoding='utf-8') as f:
        gold_data = [json.loads(line) for line in f if line.strip()]
    
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = [json.loads(line) for line in f if line.strip()]
    
    total = len(gold_data)
    correct = 0
    recall_5_sum = 0
    mrr_5_sum = 0
    
    # 收集所有的答案对用于批量计算BERT Score
    gold_answers = [gold['answer'].lower() for gold in gold_data]
    pred_answers = [pred['answer'].lower() for pred in pred_data]
    
    # 计算BERT Score
    P, R, F1 = score(pred_answers, gold_answers, lang="en", verbose=False)
    # 转换为列表
    bert_scores = F1.tolist()
    
    for i, (gold, pred) in enumerate(zip(gold_data, pred_data)):
        if gold['answer'].lower() == pred['answer'].lower():
            correct += 1

        true_doc_id = gold['document_id']
        pred_doc_ids = pred['document_id']
        
        # Recall@5
        if true_doc_id in pred_doc_ids:
            recall_5_sum += 1
            
        # MRR@5
        if true_doc_id in pred_doc_ids:
            rank = pred_doc_ids.index(true_doc_id) + 1
            mrr_5_sum += 1.0 / rank
    
    accuracy = correct / total
    bert_score_avg = sum(bert_scores) / total
    recall_5 = recall_5_sum / total
    mrr_5 = mrr_5_sum / total
    
    return {
        'accuracy': accuracy,
        'bert_score': bert_score_avg,
        'recall@5': recall_5,
        'mrr@5': mrr_5
    }

if __name__ == "__main__":
    '''
    Please run the following command to get the evaluation results:
    python metrics_calculation.py
    
    val.jsonl: the gold file
    Format:
    {
        "question": str,
        "answer": str,
        "document_id": int
    }
    
    val_predict.jsonl: the pred file
    Format:
    {
        "question": str,
        "answer": str,
        "document_id": list[int]
    }
    
    The evaluation results will be printed in the console.
    
    Note: We will use the test part for evaluating your system. You have to provide the pred file for the test part, named as 'test_predict.jsonl'.
    Format:
    {
        "question": str,
        "answer": str,
        "document_id": list[int]
    }
    '''
    gold_file_name = 'data/val.jsonl'
    pred_file_name = 'data/val_predict.jsonl'
    
    metrics = calculate_metrics(gold_file_name, pred_file_name)
    print(f"Evaluation Result:", flush=True)
    print(f"Answer Accuracy:             {metrics['accuracy']:.4f}", flush=True)
    print(f"Answer BERT Score:           {metrics['bert_score']:.4f}", flush=True)
    print(f"Document Retrieval Recall@5: {metrics['recall@5']:.4f}", flush=True)
    print(f"Document Retrieval MRR@5   : {metrics['mrr@5']:.4f}", flush=True)
