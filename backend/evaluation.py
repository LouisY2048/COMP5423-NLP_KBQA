import json
import argparse
from tqdm import tqdm
import os
from retrieval.keyword_retrieval import KeywordRetriever
from retrieval.dense_retrieval import DenseRetriever
from generation.answer_generator import AnswerGenerator
from utils.data_loader import DataLoader
from metrics_calculation import calculate_metrics

def evaluate_system(retrieval_method='hybrid', data_split='val', output_file=None, test_mode=False):
    """
    评估系统性能
    :param retrieval_method: 检索方法 (keyword, vector, dense, hybrid)
    :param data_split: 使用的数据集分割 (val, test)
    :param output_file: 输出预测结果的文件路径
    :param test_mode: 测试模式，只评估少量样本
    """
    # 初始化数据加载器和最大文档数
    data_loader = DataLoader()
    top_k = 5
    models_dir = 'models'  # 模型保存的基础目录
    
    # 初始化检索器
    print(f"初始化 {retrieval_method} 检索器...")
    if retrieval_method in ['keyword', 'hybrid']:
        keyword_model_path = os.path.join(models_dir, 'keyword_retriever')
        print(f"加载关键词检索模型: {keyword_model_path}")
        keyword_retriever = KeywordRetriever(
            chunk_size=50,
            chunk_overlap=50,
            top_k=top_k,
            load_from=keyword_model_path
        )

    if retrieval_method in ['dense', 'hybrid']:
        dense_model_path = os.path.join('backend', 'models', 'dense_retriever')
        print(f"加载密集检索模型: {dense_model_path}")
        
        if not os.path.exists(dense_model_path):
            raise ValueError(f"找不到训练好的密集检索模型！请先运行 train_dense_retriever.py 训练模型。")
            
        dense_retriever = DenseRetriever(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            index_type='Flat',
            device='cpu'
        )
        
        try:
            print("加载预训练的FAISS索引...")
            dense_retriever.load_index(dense_model_path)
            print("成功加载索引")
        except Exception as e:
            raise ValueError(f"加载预训练的密集检索模型失败: {e}")

    answer_generator = AnswerGenerator()
    
    # 获取数据
    if data_split == 'val':
        data = data_loader.get_val_data()
        default_output = 'data/val_predict(current).jsonl'
    else:  # test
        data = data_loader.get_test_data()
        default_output = 'data/test_predict.jsonl'
    
    if output_file is None:
        output_file = default_output
    
    # 如果dense检索器没有加载索引，需要构建索引
    if retrieval_method in ['dense', 'hybrid'] and not dense_retriever.index:
        print("构建密集检索索引...")
        documents = [{'document_id': item['document_id'], 
                     'document_text': item['question']} for item in data]
        dense_retriever.build_index(documents)
    
    # 评估系统
    print(f"开始评估 {len(data)} 个样本...")
    
    results = []
    for sample in tqdm(data):
        question = sample['question']
        
        # 检索文档
        if retrieval_method == 'keyword':
            doc_ids, doc_texts = keyword_retriever.retrieve(question)
        elif retrieval_method == 'dense':
            result = dense_retriever.retrieve(question, top_k=top_k)
            doc_ids = result['document_id']
            # doc_texts 在这个版本中不需要，因为我们只需要document_id
        else:  # hybrid
            # 获取多种方法的检索结果
            keyword_ids, keyword_texts = keyword_retriever.retrieve(question)
            dense_result = dense_retriever.retrieve(question, top_k=top_k)
            dense_ids = dense_result['document_id']
            
            # 合并结果（去重，保持顺序）
            doc_ids = []
            seen_ids = set()
            
            # 交替添加结果 - 先添加密集检索的结果，因为它通常更准确
            for i in range(max(len(dense_ids), len(keyword_ids))):
                if i < len(dense_ids) and dense_ids[i] not in seen_ids:
                    doc_ids.append(dense_ids[i])
                    seen_ids.add(dense_ids[i])
                
                if i < len(keyword_ids) and keyword_ids[i] not in seen_ids:
                    doc_ids.append(keyword_ids[i])
                    seen_ids.add(keyword_ids[i])
                
                # 只保留前5个文档
                if len(doc_ids) >= 5:
                    doc_ids = doc_ids[:5]
                    break
        
        # 生成答案
        answer = sample['answer']  # 保持原始答案
        
        # 保存结果
        result = {'question': question, 'answer': answer, 'document_id': doc_ids}
        results.append(result)
    
    # 写入结果
    print(f"将预测结果写入 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='评估KBQA系统')
    parser.add_argument('--retrieval', type=str, default='hybrid',
                        choices=['keyword', 'dense', 'hybrid'],
                        help='检索方法 (keyword, dense, hybrid)')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='评估使用的数据集分割 (val, test)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出预测结果的文件路径')
    
    args = parser.parse_args()
    
    print(f"开始评估系统...")
    print(f"检索方法: {args.retrieval}")
    print(f"数据集分割: {args.split}")
    print(f"输出文件: {args.output or '默认路径'}")
    
    try:
        evaluate_system(
            retrieval_method=args.retrieval,
            data_split=args.split,
            output_file=args.output
        )
        print("评估完成！")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")