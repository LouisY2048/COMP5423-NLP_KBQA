"""Base retriever class that defines the interface for all retrievers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import os
import json

class BaseRetriever(ABC):
    """Abstract base class for all retrievers"""
    
    def __init__(self, top_k=5, chunk_size=50, chunk_overlap=20):
        """Initialize the retriever"""
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Retrieve relevant documents for the query
        
        Args:
            query: Query text
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing retrieved results
        """
        # 获取文档级别的得分
        doc_scores = self._get_doc_scores(query)
        
        # 获取top-k文档
        top_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top_doc_ids = [doc_id for doc_id, _ in top_doc_ids]
        #print(len(top_doc_ids))
        # # 获取这些文档的文本
        # top_doc_texts = [self.doc_texts[self.doc_ids.index(doc_id)] for doc_id in top_doc_ids]
        
        # # 存储每个文档的块及其得分
        # doc_chunks = {}  # {doc_id: [(chunk_text, score), ...]}
        
        # # 对每个文档进行分块并计算得分
        # for doc_id, doc_text in zip(top_doc_ids, top_doc_texts):
        #     chunks = self._chunk_text(doc_text)
        #     chunk_scores = self._get_chunk_scores(question, chunks)
            
        #     # 将该文档的所有块和得分存储起来
        #     doc_chunks[doc_id] = list(zip(chunks, chunk_scores))
        
        # 对每个文档的块按得分排序，并取前5个
        result_doc_ids = []
        result_doc_texts = []
        
        for doc_id in top_doc_ids:
            # # 按得分排序该文档的所有块
            # sorted_chunks = sorted(doc_chunks[doc_id], key=lambda x: x[1], reverse=True)
            # top_chunks = sorted_chunks[:5]  # 取前5个块
            
            # 将文档ID和文本添加到结果中
            result_doc_ids.append(doc_id)
            
            # # 按指定格式组合前5个块
            # chunk_texts = []
            # for i, (chunk_text, _) in enumerate(top_chunks, 1):
            #     chunk_texts.append(f"chunk {i}: {chunk_text}")
            # combined_text = "\n".join(chunk_texts)
            
            # result_doc_texts.append(combined_text)
        
        return result_doc_ids, result_doc_texts
    
    def _get_doc_scores(self, question):
        """获取每个文档对问题的相关性得分（由子类实现）"""
        raise NotImplementedError
    
    def _get_chunk_scores(self, question, chunks):
        """获取每个块对问题的相关性得分（由子类实现）"""
        raise NotImplementedError
    
    def save_model(self, path):
        """保存模型的基本实现（由子类扩展）"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 保存基本配置
        config = {
            'top_k': self.top_k,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'total_chunks': len(self.chunks)
        }
        
        # 保存分块信息
        chunks_info = {
            'doc_ids': self.doc_ids,
            'chunks': self.chunks,
            'chunk_doc_ids': self.chunk_doc_ids,
            'chunk_indices': self.chunk_indices,
            'doc_to_chunks': self.doc_to_chunks
        }
        
        return config, chunks_info
    
    def _load_saved_data(self, path):
        """加载保存的数据的基本实现（由子类扩展）"""
        # 加载配置
        config_path = os.path.join(path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.chunk_size = config['chunk_size']
                self.chunk_overlap = config['chunk_overlap']
                self.top_k = config['top_k']
        
        # 加载分块信息
        chunks_path = os.path.join(path, 'chunks_info.json')
        if os.path.exists(chunks_path):
            with open(chunks_path, 'r') as f:
                chunks_info = json.load(f)
                self.doc_ids = chunks_info['doc_ids']
                self.chunks = chunks_info['chunks']
                self.chunk_doc_ids = chunks_info['chunk_doc_ids']
                self.chunk_indices = chunks_info['chunk_indices']
                self.doc_to_chunks = {int(k): v for k, v in chunks_info['doc_to_chunks'].items()} 