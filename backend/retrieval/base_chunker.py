import os
from nltk.tokenize import word_tokenize, sent_tokenize

class BaseChunker:
    """文档分块的基类，提供通用的分块功能"""
    
    def __init__(self, chunk_size=50, chunk_overlap=50):
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        
        # 用于存储分块信息
        self.doc_ids = []
        self.doc_texts = []
        self.chunks = []
        self.chunk_doc_ids = []
        self.chunk_indices = []
        self.doc_to_chunks = {}
    
    def _chunk_text(self, text):
        """将文本分成重叠的块"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        try:
            for sentence in sentences:
                words = word_tokenize(sentence)
                sentence_length = len(words)
                
                if current_length + sentence_length <= self.chunk_size or not current_chunk:
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    chunks.append(' '.join(current_chunk))
                    
                    overlap_start = max(0, (len(current_chunk) - self.chunk_overlap))
                    current_chunk = current_chunk[overlap_start:]
                    current_length = sum(len(word_tokenize(s)) for s in current_chunk)
                    
                    current_chunk.append(sentence)
                    current_length += sentence_length
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            if not chunks:
                chunks = [text]
                
            return chunks
        except Exception as e:
            print(f"分块过程出错: {str(e)}")
            raise e
    
    def process_documents(self, documents):
        """处理文档并创建分块"""
        # 获取已处理的文档ID
        processed_doc_ids = set(self.doc_ids)
        
        # 计算需要处理的文档
        docs_to_process = {
            doc_id: doc_text for doc_id, doc_text in documents.items()
            if doc_id not in processed_doc_ids
        }
        
        try:
            # 处理剩余文档
            for doc_id, doc_text in docs_to_process.items():
                self.doc_ids.append(doc_id)
                self.doc_texts.append(doc_text)
                
                # 分块文档
                doc_chunks = self._chunk_text(doc_text)
                self.doc_to_chunks[doc_id] = []
                
                # 存储块信息
                for j, chunk in enumerate(doc_chunks):
                    self.chunks.append(chunk)
                    self.chunk_doc_ids.append(doc_id)
                    self.chunk_indices.append(j)
                    self.doc_to_chunks[doc_id].append(len(self.chunks) - 1)
            
        except Exception as e:
            print(f"\n处理文档时发生错误: {str(e)}")
            raise e
    
    def get_model_specific_info(self):
        """获取模型特定的信息，由子类实现"""
        return {} 