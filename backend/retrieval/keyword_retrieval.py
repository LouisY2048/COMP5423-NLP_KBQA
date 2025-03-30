import os
import json
import re
import nltk
import numpy as np
import pickle
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from utils.data_loader import DataLoader
from tqdm import tqdm
from .base_retriever import BaseRetriever

class KeywordRetriever(BaseRetriever):
    def __init__(self, chunk_size, chunk_overlap, top_k=5, k1=1.5, b=0.75, load_from=None):
        super().__init__(top_k, chunk_size, chunk_overlap)
        # 确保必要的NLTK数据可用
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # 初始化停用词列表
        self.stop_words = set(stopwords.words('english')) # 暂时只用英文停用词
        
        self.k1 = k1
        self.b = b
        self.data_loader = DataLoader()
        self.tokenized_docs = []
        
        if load_from and os.path.exists(load_from):
            print(f"从 {load_from} 加载BM25检索器...")
            self.load_model(load_from)
        else:
            print("初始化BM25检索器...")
            self._initialize_bm25()
            print("BM25检索器初始化完成")
    
    def _initialize_bm25(self):
        """初始化BM25模型"""
        # 获取所有文档
        documents = self.data_loader.get_all_documents()
        
        # 存储文档信息
        self.doc_ids = list(documents.keys())
        self.doc_texts = list(documents.values())
        
        # 预处理所有文档
        print("预处理文档...")
        self.tokenized_docs = [self._preprocess_text(doc_text) for doc_text in self.doc_texts]
        
        # 创建BM25模型
        print("创建BM25模型...")
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
        print("BM25模型初始化完成")
    
    def _preprocess_text(self, text):
        """预处理文本"""
        # 转小写
        text = text.lower()
        # 移除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text)
        # 分词
        tokens = word_tokenize(text)
        # 移除停用词
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def _get_doc_scores(self, question):
        """获取每个文档对问题的相关性得分"""
        # 预处理问题
        query_tokens = self._preprocess_text(question)
        
        # 计算BM25得分
        scores = self.bm25.get_scores([query_tokens])[0]
        print("2222222222222222")
        return {doc_id: score for doc_id, score in zip(self.doc_ids, scores)}
    
    def _get_chunk_scores(self, question, chunks):
        """获取每个块对问题的相关性得分"""
        # 预处理问题
        query_tokens = self._preprocess_text(question)
        
        # 预处理所有块
        chunk_tokens = [self._preprocess_text(chunk) for chunk in chunks]
        
        # 计算BM25得分
        scores = self.bm25.get_scores(chunk_tokens)
        return scores
    
    def train(self, training_data=None, k1_range=[1.0, 1.2, 1.5, 1.8, 2.0], 
              b_range=[0.5, 0.65, 0.75, 0.85, 0.95]):
        """
        通过网格搜索优化BM25参数
        
        Args:
            training_data: 带有问题和正确文档ID的训练数据
            k1_range: k1参数的候选值范围
            b_range: b参数的候选值范围
        
        Returns:
            最佳参数
        """
        if training_data is None:
            print("没有提供训练数据，使用默认参数")
            return {'k1': self.k1, 'b': self.b}
        
        print("开始优化BM25参数...")
        best_recall = 0
        best_params = {'k1': self.k1, 'b': self.b}
        
        # 准备文档集合
        if not self.tokenized_docs:
            print("准备文档集合...")
            documents = self.data_loader.get_all_documents()
            for doc_id, doc_text in tqdm(documents.items(), desc="处理文档", total=len(documents)):
                self.doc_ids.append(doc_id)
                self.doc_texts.append(doc_text)
                self.tokenized_docs.append(self._preprocess_text(doc_text))
        
        # 计算总参数组合数
        total_combinations = len(k1_range) * len(b_range)
        print(f"将测试 {total_combinations} 种参数组合...")
        
        # 创建参数组合列表，用于进度条
        param_combinations = [(k1, b) for k1 in k1_range for b in b_range]
        
        # 使用tqdm显示总体训练进度
        for i, (k1, b) in enumerate(tqdm(param_combinations, desc="参数网格搜索", total=total_combinations)):
            print(f"\n[{i+1}/{total_combinations}] 测试参数: k1={k1}, b={b}")
            
            # 使用当前参数创建BM25模型
            test_bm25 = BM25Okapi(self.tokenized_docs, k1=k1, b=b)
            
            # 在训练数据上评估
            recall_sum = 0
            # 添加评估进度条
            evaluation_progress = tqdm(training_data, desc=f"评估 k1={k1}, b={b}", total=len(training_data))
            for sample in evaluation_progress:
                question = sample['question']
                gold_doc_ids = sample['document_id']
                
                # 检索文档
                query_tokens = self._preprocess_text(question)
                doc_scores = test_bm25.get_scores(query_tokens)
                top_indices = np.argsort(doc_scores)[::-1][:self.top_k]
                retrieved_doc_ids = [self.doc_ids[idx] for idx in top_indices]
                
                # 计算召回率
                hits = sum(1 for doc_id in gold_doc_ids if doc_id in retrieved_doc_ids)
                recall = hits / len(gold_doc_ids) if gold_doc_ids else 0
                recall_sum += recall
                
                # 更新评估进度条显示当前召回率
                evaluation_progress.set_postfix({'当前召回率': f"{recall:.4f}"})
            
            avg_recall = recall_sum / len(training_data)
            print(f"参数 k1={k1}, b={b} 的平均召回率@{self.top_k}: {avg_recall:.4f}")
            
            if avg_recall > best_recall:
                best_recall = avg_recall
                best_params = {'k1': k1, 'b': b}
                print(f"✓ 发现更好的参数: {best_params}, 召回率: {best_recall:.4f}")
        
        # 使用最佳参数更新模型
        print(f"\n训练完成！最佳参数: {best_params}, 最佳召回率: {best_recall:.4f}")
        self.k1 = best_params['k1']
        self.b = best_params['b']
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
        
        return best_params
    
    def save_model(self, path='../models/bm25_model.pkl'):
        """保存模型到文件"""
        # 获取基本配置和分块信息
        config, chunks_info = super().save_model(path)
        
        # 准备模型特定的数据
        model_data = {
            'k1': self.k1,
            'b': self.b,
            'tokenized_docs': self.tokenized_docs,
            **config,
            **chunks_info
        }
        
        # 保存数据
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"BM25模型及文档块已保存到 {path}")
    
    def load_model(self, path):
        """从文件加载模型"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        # 加载基本数据
        self.doc_ids = model_data['doc_ids']
        self.doc_texts = model_data['doc_texts']
        self.tokenized_docs = model_data['tokenized_docs']
        
        if 'chunk_size' in model_data:
            self.chunk_size = model_data['chunk_size']
        if 'chunk_overlap' in model_data:
            self.chunk_overlap = model_data['chunk_overlap']
        
        # 重建BM25模型
        self.bm25 = BM25Okapi(self.tokenized_docs, k1=self.k1, b=self.b)
        
        print(f"BM25模型及文档块已从 {path} 加载")