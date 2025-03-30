"""
训练模型脚本，用于训练和保存KBQA系统中的检索模型
支持训练:
1. Dense Retrieval模型 (基于BERT的语义检索)
2. Keyword Retrieval模型 (基于BM25的关键词检索)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from retrieval.dense_retrieval import DenseRetriever
from retrieval.keyword_retrieval import KeywordRetriever

"""
python train_model.py --model dense
python train_model.py --model keyword
"""

def load_training_data(file_path):
    """
    加载训练数据
    
    Args:
        file_path: 训练数据文件路径
        
    Returns:
        训练数据列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                data.append(sample)
        print(f"成功加载 {len(data)} 条训练数据")
        return data
    except Exception as e:
        print(f"加载训练数据出错: {e}")
        sys.exit(1)

def train_dense_retriever(args):
    """训练Dense Retrieval模型"""
    print("\n" + "="*50)
    print(f"开始训练Dense Retrieval模型")
    print("="*50)
    
    # 加载训练数据
    train_data = load_training_data(args.train_file)
    
    # 配置固定的输出路径
    output_path = os.path.join(args.output_dir, "dense_retriever")
    os.makedirs(output_path, exist_ok=True)
    
    # 初始化模型
    print(f"初始化Dense Retriever (chunk_size: {args.chunk_size})")
    retriever = DenseRetriever(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        word2vec_path="retrieval/pre-trained-models/GoogleNews-vectors-negative300.bin"
    )
    
    # 训练模型
    print("\n开始训练...")
    retriever.train(
        training_data=train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        output_path=output_path
    )
    
    print(f"\n训练完成！模型已保存到: {output_path}")
    
    # 保存训练配置
    config = {
        'model_type': 'dense',
        'train_file': args.train_file,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'top_k': args.top_k,
        'train_samples': len(train_data),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    with open(os.path.join(output_path, 'training_config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"训练配置已保存到: {os.path.join(output_path, 'training_config.json')}")
    
    return output_path

def train_keyword_retriever(args):
    """训练Keyword Retrieval模型（BM25参数优化）"""
    print("\n" + "="*50)
    print(f"开始训练Keyword Retrieval模型（BM25参数优化）")
    print("="*50)
    
    # 加载训练数据
    train_data = load_training_data(args.train_file)
    
    # 配置固定的输出路径
    output_path = os.path.join(args.output_dir, "keyword_retriever")
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, 'bm25_model.pkl')
    
    # 初始化模型
    print(f"初始化Keyword Retriever (chunk_size: {args.chunk_size})")
    retriever = KeywordRetriever(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k
    )
    
    # 训练模型（优化参数）
    print("\n开始训练(参数优化)...")
    best_params = retriever.train(
        training_data=train_data,
        k1_range=args.k1_range,
        b_range=args.b_range
    )
    
    # 保存模型
    retriever.save_model(model_path)
    print(f"\n训练完成！模型已保存到: {model_path}")
    
    # 保存训练配置
    config_path = os.path.join(output_path, 'bm25_training_config.json')
    config = {
        'model_type': 'keyword_bm25',
        'train_file': args.train_file,
        'best_params': best_params,
        'k1_range': args.k1_range,
        'b_range': args.b_range,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'top_k': args.top_k,
        'train_samples': len(train_data),
        'model_path': model_path
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"训练配置已保存到: {config_path}")
    
    return output_path

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练KBQA系统检索模型')
    
    # 基本参数
    parser.add_argument('--model', type=str, required=True, choices=['dense', 'keyword'],
                       help='要训练的模型类型: dense(密集向量检索) 或 keyword(关键词检索)')
    parser.add_argument('--train_file', type=str, default='../data/train.jsonl',
                       help='训练数据文件路径')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='模型输出目录')
    parser.add_argument('--top_k', type=int, default=5,
                       help='检索时返回的文档数量')
    
    # 文本分块参数
    parser.add_argument('--chunk_size', type=int, default=50,
                       help='文本分块大小(单词数)')
    parser.add_argument('--chunk_overlap', type=int, default=50,
                       help='文本分块重叠大小(单词数)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批处理大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='预热步数比例')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 检查输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"创建输出目录: {args.output_dir}")
    
    # 基于选定的模型类型进行训练
    if args.model == 'dense':
        output_path = train_dense_retriever(args)
    elif args.model == 'keyword':
        output_path = train_keyword_retriever(args)
    
    print("\n" + "="*50)
    print(f"模型训练和保存完成！")
    print(f"模型路径: {output_path}")
    print("="*50)

if __name__ == "__main__":
    main() 