from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from contextlib import asynccontextmanager

from retrieval.keyword_retrieval import KeywordRetriever
from retrieval.dense_retrieval import DenseRetriever
from generation.answer_generator import AnswerGenerator

# 启动命令
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 初始化检索器和生成器
keyword_retriever = None
dense_retriever = None
answer_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    global keyword_retriever, dense_retriever, answer_generator
    print("Loading retrieval models...")
    keyword_retriever = KeywordRetriever(
        chunk_size=50,
        chunk_overlap=50,
        top_k=5
    )
    dense_retriever = DenseRetriever(
        chunk_size=50,
        chunk_overlap=50,
        top_k=5,
        word2vec_path="retrieval/pre-trained-models/GoogleNews-vectors-negative300.bin"
    )
    print("Loading answer generation model...")
    answer_generator = AnswerGenerator()
    print("System ready!")
    
    yield  # 此处暂停，直到应用关闭
    
    # 关闭时执行
    print("Shutting down...")

app = FastAPI(title="KBQA System API", lifespan=lifespan)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    retrieval_method: str = "hybrid"  # keyword, vector, dense, hybrid

class AnswerResponse(BaseModel):
    question: str
    answer: str
    document_id: List[int]
    documents: List[str]

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # 根据选择的检索方法获取相关文档
        if request.retrieval_method == "keyword":
            doc_ids, doc_texts = keyword_retriever.retrieve(request.question)
        elif request.retrieval_method == "dense":
            doc_ids, doc_texts = dense_retriever.retrieve(request.question)
        elif request.retrieval_method == "hybrid":
            # 结合所有检索方法
            keyword_ids, keyword_texts = keyword_retriever.retrieve(request.question)
            dense_ids, dense_texts = dense_retriever.retrieve(request.question)
            
            # 简单合并去重，保持顺序
            doc_ids = []
            doc_texts = []
            seen_ids = set()
            
            # 交替添加结果 - 先添加密集检索的结果，因为它通常更准确
            for i in range(max(len(dense_ids), len(keyword_ids))):
                if i < len(dense_ids) and dense_ids[i] not in seen_ids:
                    doc_ids.append(dense_ids[i])
                    doc_texts.append(dense_texts[i])
                    seen_ids.add(dense_ids[i])
                
                if i < len(keyword_ids) and keyword_ids[i] not in seen_ids:
                    doc_ids.append(keyword_ids[i])
                    doc_texts.append(keyword_texts[i])
                    seen_ids.add(keyword_ids[i])
                
                # 只保留前5个文档
                if len(doc_ids) >= 5:
                    doc_ids = doc_ids[:5]
                    doc_texts = doc_texts[:5]
                    break
        else:
            raise HTTPException(status_code=400, detail="Invalid retrieval method")
        
        # 生成答案
        answer = answer_generator.generate(request.question, doc_texts, generate_type="user")
        
        return {
            "question": request.question,
            "answer": answer,
            "document_id": doc_ids,
            "documents": doc_texts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 