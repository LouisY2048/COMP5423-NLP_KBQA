import os
import json
import re
from bs4 import BeautifulSoup
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from nltk import sent_tokenize

class DocumentModifier:
    def __init__(self):
        self.project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # 检查GPU是否可用
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # 预编译正则表达式以提高性能
        self.whitespace_re = re.compile(r'\s+')
        self.control_chars_re = re.compile(r'[\x00-\x1F\x7F-\x9F]')
        self.url_re = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_re = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.punct_re = re.compile(r'[^\w\s.,!?-]')
        self.repeat_punct_re = re.compile(r'([.,!?])\1+')
        self.stopwords_re = re.compile(r'\b(the|and|is|in|to|of|a|that|it|with|as|for|by|on|at|from|up|down|out|about|into|over|under|again|further|then|once|here|there|when|where|why|how|all|any|some|one|two|three|four|five|six|seven|eight|nine|ten)\b')
    
    def clean_text(self, text):
        """清理文本中的HTML标签和其他格式化内容"""
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(text, 'html.parser')
        # 删除格式为二级标题的"Contents"及其之后的内容，直到下一个二级标题
        contents_h2 = soup.find('h2', string=lambda text: text and 'Contents' in text)
        if contents_h2:
            # 找到下一个h2标签
            next_h2 = contents_h2.find_next('h2')
            if next_h2:
                # 删除从Contents h2到下一个h2之间的所有内容
                current = contents_h2
                while current and current != next_h2:
                    next_sibling = current.next_sibling
                    current.decompose()
                    current = next_sibling
            else:
                # 如果没有下一个h2，删除Contents及其之后的所有内容
                current = contents_h2
                while current:
                    next_sibling = current.next_sibling
                    current.decompose()
                    current = next_sibling
        # 提供一个列表，包含需要删除的二级标题
        delete_h2_list = ['See to', 'See also']
        
        # 删除格式为二级标题的delete_h2_list之中的内容，以及其后的所有内容
        for h2 in soup.find_all('h2'):
            h2_text = h2.get_text()
            if any(keyword in h2_text for keyword in delete_h2_list):
                current = h2
                while current:
                    next_sibling = current.next_sibling
                    current.decompose()
                    current = next_sibling
        
        # 最后才将HTML转换为纯文本
        text = soup.get_text()
        
        # 清理多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 清理特殊字符和控制字符
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # 清理URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # 清理多余的标点符号
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        
        # 清理开头和结尾的空白字符
        text = text.strip()
        
        return text
    
    def clean_batch(self, texts, batch_size=32):
        """批量清理文本"""
        if self.device == 'cuda':
            results = []
            # 使用线程池进行并行处理
            with ThreadPoolExecutor(max_workers=4) as executor:
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    # 并行处理每个batch中的文本
                    batch_results = list(executor.map(self.clean_text, batch))
                    results.extend(batch_results)
            return results
        else:
            return [self.clean_text(text) for text in texts]
    
    def split_documents(self, input_file=None, output_file=None, chunk_size=500, chunk_overlap=0):
        """将文档分块并保存
        Args:
            input_file: 输入文件路径，默认为'data/documents_modified.jsonl'
            output_file: 输出文件路径，默认为'data/documents_splitted.jsonl'
            chunk_size: 每个块的最大字符数
            chunk_overlap: 块之间的重叠字符数
        """
        if input_file is None:
            input_file = os.path.join(self.project_dir, 'data', 'documents_modified.jsonl')
        
        if output_file is None:
            output_dir = os.path.dirname(input_file)
            output_file = os.path.join(output_dir, 'documents_splitted.jsonl')
        
        print(f"开始处理文档分块...")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        chunk_id = 0
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in tqdm(f_in, desc="处理进度"):
                doc = json.loads(line)
                doc_id = doc['document_id']
                text = doc['document_text']
                
                # 按句子分割文本
                sentences = sent_tokenize(text)
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence)
                    
                    # 如果当前句子加上已有内容超过chunk_size，保存当前chunk并开始新的chunk
                    if current_length + sentence_length > chunk_size and current_chunk:
                        # 保存当前chunk
                        chunk_text = ' '.join(current_chunk)
                        chunk_data = {
                            "chunk_id": chunk_id,
                            "document_id": doc_id,
                            "chunk_text": chunk_text
                        }
                        f_out.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
                        chunk_id += 1
                        
                        # 开始新的chunk
                        current_chunk = []
                        current_length = 0
                    
                    current_chunk.append(sentence)
                    current_length += sentence_length
                
                # 处理最后一个chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "chunk_text": chunk_text
                    }
                    f_out.write(json.dumps(chunk_data, ensure_ascii=False) + '\n')
                    chunk_id += 1
        
        print(f"文档分块完成！")
        print(f"共生成 {chunk_id} 个文本块")
        print(f"分块结果已保存至: {output_file}")

    def process_documents(self, input_file=None, output_file=None):
        """处理文档文件"""
        if input_file is None:
            input_file = os.path.join(self.project_dir, 'data', 'documents.jsonl')
        
        if output_file is None:
            output_dir = os.path.dirname(input_file)
            output_file = os.path.join(output_dir, 'documents_modified.jsonl')
        
        print(f"开始处理文档...")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 读取并处理文档
        modified_docs = []
        total_docs = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
        removed_content_count = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_docs, desc="处理进度"):
                doc = json.loads(line)
                
                # 清理文本
                if 'document_text' in doc:
                    original_length = len(doc['document_text'])
                    doc['document_text'] = self.clean_text(doc['document_text'])
                    # 统计删除了"See to"部分的文档数量
                    if len(doc['document_text']) < original_length:
                        removed_content_count += 1
                
                modified_docs.append(doc)
        
        # 保存处理后的文档
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in modified_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        print(f"处理完成！共处理了 {len(modified_docs)} 个文档")
        print(f"其中 {removed_content_count} 个文档删除了'See to'相关内容")
        print(f"处理后的文档已保存至: {output_file}")
        
        # 进行文档分块
        print("\n开始进行文档分块...")
        self.split_documents(output_file)

def main():
    modifier = DocumentModifier()
    # modifier.process_documents()
    modifier.split_documents()

if __name__ == "__main__":
    main() 