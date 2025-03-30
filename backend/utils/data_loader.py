import json
import os

class DataLoader:
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.documents = {}
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        # 加载文档数据
        self._load_documents()
        
        # 加载训练、验证和测试数据
        self._load_train_data()
        self._load_val_data()
        self._load_test_data()
        
        print(f"加载完成: {len(self.documents)} 文档, {len(self.train_data)} 训练样本, "
              f"{len(self.val_data)} 验证样本, {len(self.test_data)} 测试样本")
    
    def _load_documents(self):
        """加载文档数据"""
        doc_path = os.path.join(self.data_dir, 'documents.jsonl')
        print(f"加载文档数据从 {doc_path}")
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                doc = json.loads(line)
                self.documents[doc['document_id']] = doc['document_text']
    
    def _load_train_data(self):
        """加载训练数据"""
        train_path = os.path.join(self.data_dir, 'train.jsonl')
        
        if os.path.exists(train_path):
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.train_data.append(json.loads(line))
    
    def _load_val_data(self):
        """加载验证数据"""
        val_path = os.path.join(self.data_dir, 'val.jsonl')
        
        if os.path.exists(val_path):
            with open(val_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.val_data.append(json.loads(line))
    
    def _load_test_data(self):
        """加载测试数据"""
        test_path = os.path.join(self.data_dir, 'test.jsonl')
        
        if os.path.exists(test_path):
            with open(test_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    self.test_data.append(json.loads(line))
    
    def get_document_by_id(self, doc_id):
        """根据ID获取文档内容"""
        return self.documents.get(doc_id, "")
    
    def get_all_documents(self):
        """获取所有文档"""
        return self.documents
    
    def get_document_ids(self):
        """获取所有文档ID"""
        return list(self.documents.keys())
    
    def get_train_data(self):
        """获取训练数据"""
        return self.train_data
    
    def get_val_data(self):
        """获取验证数据"""
        return self.val_data
    
    def get_test_data(self):
        """获取测试数据"""
        return self.test_data 