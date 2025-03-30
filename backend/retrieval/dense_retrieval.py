import os
import torch
import numpy as np
import faiss
import json
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
from .base_retriever import BaseRetriever
from utils.data_loader import DataLoader

class QADataset(Dataset):
    """问答对数据集，用于训练DPR模型"""
    def __init__(self, questions, passages, labels, vector_dim=300):
        self.questions = questions
        self.passages = passages
        self.labels = labels
        self.vector_dim = vector_dim
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        passage = self.passages[idx]
        label = self.labels[idx]
        
        return {
            'question': question,
            'passage': passage,
            'label': torch.tensor(label, dtype=torch.float)
        }

class DenseRetriever(BaseRetriever):
    """
    密集通道检索(Dense Passage Retrieval, DPR)实现
    使用预训练的Word2Vec模型作为编码器，将问题和文档编码为密集向量
    """
    def __init__(self, 
                 chunk_size,
                 chunk_overlap,
                 word2vec_path="pre-trained-models/GoogleNews-vectors-negative300.bin",
                 top_k=5,
                 load_from=None):
        super().__init__(top_k, chunk_size, chunk_overlap)
        
        # 初始化数据加载器
        self.data_loader = DataLoader()
        
        # 初始化检查点目录
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 加载Word2Vec模型
        print(f"加载Word2Vec模型: {word2vec_path}")
        self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        self.vector_dim = self.word2vec.vector_size
        print(f"Word2Vec模型加载完成，向量维度: {self.vector_dim}")
        
        if load_from and os.path.exists(load_from):
            self._load_saved_data(load_from)
        else:
            print("初始化检索器...")
        self._build_index()
        print("检索器初始化完成")
    
    def _encode_text(self, text):
        """将文本编码为向量（使用Word2Vec词向量的平均值）"""
        # 分词并移除标点
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalnum()]
        
        # 获取词向量并计算平均值
        vectors = []
        for word in words:
            if word in self.word2vec:
                vectors.append(self.word2vec[word])
        
        if vectors:
            # 返回所有词向量的平均值
            return np.mean(vectors, axis=0)
        else:
            # 如果没有找到任何词向量，返回零向量
            return np.zeros(self.vector_dim)
    
    def _encode_batch(self, texts, batch_size=32):
        """批量编码文本"""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="编码文本"):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = [self._encode_text(text) for text in batch_texts]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _build_index(self):
        """构建FAISS索引"""
        # 检查是否存在已保存的索引
        index_path = os.path.join(self.checkpoint_dir, 'faiss_index.bin')
        if os.path.exists(index_path):
            try:
                print("发现已有索引，正在加载...")
                self.index = faiss.read_index(index_path)
                print(f"已加载FAISS索引，包含 {self.index.ntotal} 个向量")
                
                # 获取所有文档以保持文档ID和文本的对应关系
                documents = self.data_loader.get_all_documents()
                self.doc_ids = list(documents.keys())
                self.doc_texts = list(documents.values())
                
                # 从索引中获取文档嵌入
                self.doc_embeddings = np.zeros((self.index.ntotal, self.vector_dim), dtype='float32')
                self.index.reconstruct_n(0, self.index.ntotal, self.doc_embeddings)
                
                return
            except Exception as e:
                print(f"加载索引时出错: {str(e)}")
                print("将重新构建索引...")
        
        # 如果没有找到索引或加载失败，重新构建
        print("构建新索引...")
        documents = self.data_loader.get_all_documents()
        
        # 存储文档信息
        self.doc_ids = list(documents.keys())
        self.doc_texts = list(documents.values())
        
        # 编码所有文档
        print("编码文档...")
        self.doc_embeddings = self._encode_batch(self.doc_texts)
        
        # 创建FAISS索引
        print("创建FAISS索引...")
        try:
            self.index = faiss.IndexFlatIP(self.vector_dim)
            self.index.add(self.doc_embeddings.astype('float32'))
            
            # 保存FAISS索引
            faiss.write_index(self.index, index_path)
            print("索引构建完成并已保存")
            
        except Exception as e:
            print(f"\n创建索引时出错: {str(e)}")
            raise e
    
    def _get_doc_scores(self, question):
        """获取每个文档对问题的相关性得分"""
        # 编码问题
        query_embedding = self._encode_text(question)
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        
        # 如果有训练好的投影层，使用它来转换编码
        if hasattr(self, 'projection'):
            query_embedding = self.projection(query_embedding.unsqueeze(0)).squeeze(0)
            # 将文档嵌入也通过投影层
            doc_embeddings = torch.tensor(self.doc_embeddings, dtype=torch.float32)
            doc_embeddings = self.projection(doc_embeddings)
            
            # 计算余弦相似度
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
            doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
            scores = torch.sum(query_embedding * doc_embeddings, dim=1)
            
            # 使用 detach() 分离梯度后再转换为 numpy 数组
            scores = scores.detach().numpy()
        else:
            # 如果没有训练好的投影层，使用原始的向量相似度
            query_embedding = np.array([query_embedding]).astype('float32')
            scores = np.sum(query_embedding * self.doc_embeddings, axis=1)
        
        return {doc_id: float(score) for doc_id, score in zip(self.doc_ids, scores)}
    
    def _get_chunk_scores(self, question, chunks):
        """获取每个块对问题的相关性得分"""
        # 编码问题
        query_embedding = self._encode_text(question)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # 编码所有块
        chunk_embeddings = np.array([self._encode_text(chunk) for chunk in chunks]).astype('float32')
        
        # 计算相似度分数
        scores = np.sum(query_embedding * chunk_embeddings, axis=1)
        return scores
    
    def save_model(self, path='../models/dense_retriever/'):
        """保存模型和索引"""
        os.makedirs(path, exist_ok=True)
        
        # 获取基本配置和分块信息
        config, chunks_info = super().save_model(path)
        
        # 添加模型特定的配置
        config['vector_dim'] = self.vector_dim
        
        # 保存FAISS索引
        index_path = os.path.join(path, 'faiss_index.bin')
        faiss.write_index(self.index, index_path)
        print(f"FAISS索引已保存到: {index_path}")
        
        # 保存配置和分块信息
        config_path = os.path.join(path, 'config.json')
        chunks_path = os.path.join(path, 'chunks_info.json')
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        with open(chunks_path, 'w') as f:
            json.dump(chunks_info, f)
        
        # 保存投影层参数
        if hasattr(self, 'projection'):
            projection_path = os.path.join(path, 'projection.pt')
            torch.save(self.projection.state_dict(), projection_path)
            print(f"投影层参数已保存到: {projection_path}")
    
    def _load_saved_data(self, path):
        """从保存的文件加载模型和索引"""
        # 加载基本数据
        super()._load_saved_data(path)
        
        # 加载FAISS索引
        index_path = os.path.join(path, 'faiss_index.bin')
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"已加载FAISS索引，包含 {self.index.ntotal} 个向量")
        else:
            print("未找到保存的索引，将重新构建")
            self._build_index()
        
        # 加载投影层参数
        projection_path = os.path.join(path, 'projection.pt')
        if os.path.exists(projection_path):
            self.projection = torch.nn.Linear(self.vector_dim, self.vector_dim, bias=False)
            self.projection.load_state_dict(torch.load(projection_path))
            print("已加载投影层参数")
    
    def train(self, training_data, epochs=3, batch_size=16, learning_rate=2e-5, warmup_ratio=0.1,
              output_path='../models/dense_retriever/'):
        """
        训练DPR模型
        
        Args:
            training_data: 包含问题和相关文档的训练数据
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            warmup_ratio: 预热步数比例
            output_path: 模型保存路径
        """
        print(f"开始训练DPR模型，使用 {len(training_data)} 条训练数据...")
        
        # 确保输出目录存在
        os.makedirs(output_path, exist_ok=True)
        
        # 加载所有文档用于构建训练数据
        documents = self.data_loader.get_all_documents()
        
        # 准备训练数据
        questions = []
        passages = []
        labels = []
        
        print("准备训练样本...")
        # 使用块级别的训练数据
        for sample in tqdm(training_data, desc="处理训练数据"):
            question = sample['question']
            
            # 获取正例文档
            if 'document_id' in sample and sample['document_id']:
                # 将document_id转换为列表形式
                doc_ids = [sample['document_id']] if isinstance(sample['document_id'], int) else sample['document_id']
                
                for doc_id in doc_ids:
                    if doc_id in documents:
                        # 获取该文档的所有块
                        if doc_id in self.doc_to_chunks:
                            # 使用文档的所有块作为正例
                            for chunk_idx in self.doc_to_chunks[doc_id]:
                                chunk_text = self.chunks[chunk_idx]
                                
                                # 添加正例
                                questions.append(question)
                                passages.append(chunk_text)
                                labels.append(1.0)  # 相关
                                
                                # 随机选择负例块
                                for _ in range(2):  # 每个正例配2个负例
                                    neg_chunk_idx = chunk_idx
                                    while neg_chunk_idx in self.doc_to_chunks[doc_id]:
                                        neg_chunk_idx = np.random.randint(0, len(self.chunks))
                                        
                                    questions.append(question)
                                    passages.append(self.chunks[neg_chunk_idx])
                                    labels.append(0.0)  # 不相关
                        else:
                            # 如果没有分块信息，使用整个文档
                            questions.append(question)
                            passages.append(documents[doc_id])
                            labels.append(1.0)  # 相关
                            
                            # 随机选择负例
                            for _ in range(2):  # 每个正例配2个负例
                                neg_doc_id = doc_id
                                while neg_doc_id in doc_ids:  # 修改这里以使用doc_ids
                                    neg_doc_id = np.random.choice(list(documents.keys()))
                                    
                                questions.append(question)
                                passages.append(documents[neg_doc_id])
                                labels.append(0.0)  # 不相关
        
        if len(questions) == 0:
            print("没有有效的训练样本，退出训练")
            return
            
        print(f"创建了 {len(questions)} 个训练样本")
        
        # 创建数据集和数据加载器
        train_dataset = QADataset(
            questions=questions,
            passages=passages,
            labels=labels,
            vector_dim=self.vector_dim
        )
        
        train_dataloader = TorchDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # 创建可训练的参数矩阵，用于转换编码向量
        self.projection = torch.nn.Linear(self.vector_dim, self.vector_dim, bias=False)
        optimizer = torch.optim.AdamW(self.projection.parameters(), lr=learning_rate)
        
        # 设置学习率调度器
        total_steps = len(train_dataloader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        # 简单的学习率调度器
        def get_lr(step):
            if step < warmup_steps:
                return learning_rate * (step / warmup_steps)
            return learning_rate
        
        # 设置损失函数
        loss_fn = torch.nn.BCEWithLogitsLoss()
        
        # 训练循环
        print(f"开始训练，epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
        best_loss = float('inf')
        best_epoch = 0
        current_step = 0
        
        # 创建进度条跟踪总体训练进度
        epoch_progress = tqdm(range(epochs), desc="训练进度", position=0)
        
        for epoch in epoch_progress:
            epoch_loss = 0
            batch_count = 0
            
            # 创建批次进度条
            batch_progress = tqdm(train_dataloader, 
                                  desc=f"Epoch {epoch+1}/{epochs}", 
                                  position=1, 
                                  leave=False, 
                                  total=len(train_dataloader))
            
            for batch in batch_progress:
                # 获取问题和文档
                questions = batch['question']
                passages = batch['passage']
                labels = batch['label'].to(torch.float32)
                
                # 编码问题和文档，并确保梯度可以传播
                question_embeddings = torch.stack([
                    torch.tensor(self._encode_text(q), dtype=torch.float32)
                    for q in questions
                ]).requires_grad_(True)

                passage_embeddings = torch.stack([
                    torch.tensor(self._encode_text(p), dtype=torch.float32)
                    for p in passages
                ]).requires_grad_(True)

                # 通过投影层处理编码
                question_embeddings = self.projection(question_embeddings)
                passage_embeddings = self.projection(passage_embeddings)

                # 计算余弦相似度
                norm_q = torch.nn.functional.normalize(question_embeddings, p=2, dim=1)
                norm_p = torch.nn.functional.normalize(passage_embeddings, p=2, dim=1)
                scores = torch.sum(norm_q * norm_p, dim=1)

                # 计算损失
                optimizer.zero_grad()
                loss = loss_fn(scores, labels)
                loss.backward()
                optimizer.step()
                
                # 累计损失
                epoch_loss += loss.item()
                batch_count += 1
                current_step += 1
                
                # 更新批次进度条显示
                batch_progress.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'lr': f"{get_lr(current_step):.6f}"
                })
            
            # 计算epoch平均损失
            avg_loss = epoch_loss / batch_count
            
            # 更新总进度条显示
            epoch_progress.set_postfix({
                'loss': f"{avg_loss:.4f}", 
                'best_epoch': best_epoch
            })
            
            # 判断是否为最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_epoch = epoch + 1
                
                # 保存最佳模型
                print(f"\n✓ 发现更好的模型 (Epoch {best_epoch}, Loss: {best_loss:.4f})，保存中...")
                self.save_model(output_path)
                
                # 更新总进度条
                epoch_progress.set_postfix({
                    'loss': f"{avg_loss:.4f}", 
                    'best_epoch': best_epoch
                })
        
        print(f"\n训练完成！最佳模型来自 Epoch {best_epoch}，损失: {best_loss:.4f}")
        print(f"模型已保存到: {output_path}")
        
        # 更新索引
        print("更新检索索引...")
        self._build_index()
    
