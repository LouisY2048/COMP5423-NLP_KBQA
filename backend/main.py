from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from contextlib import asynccontextmanager

from retrieval.dense_retrieval import DenseRetriever
from generation.answer_generator import AnswerGenerator

"""
# 启动命令
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""


# 初始化检索器和生成器
dense_retriever = None
answer_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行
    global dense_retriever, answer_generator
    print("Loading retrieval models...")
    dense_retriever = DenseRetriever()
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
    retrieval_method: str = "dense"

class AnswerResponse(BaseModel):
    question: str
    answer: str
    document_id: List[int]
    documents: List[str]

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    try:
        # 检查检索器是否已初始化
        if dense_retriever is None:
            raise HTTPException(status_code=500, detail="Dense retriever not initialized")
            
        # 根据选择的检索方法获取相关文档
        if request.retrieval_method == "dense":
            try:
                # 首先获取文档ID
                result = dense_retriever.retrieve(request.question)
                doc_ids = result["document_id"]
                
                # 使用answer_question_by_chunks获取文档文本
                chunks_result = dense_retriever.answer_question_by_chunks(
                    request.question,
                    retrieved_doc_ids=doc_ids
                )
                chunk_texts = chunks_result["chunks"]
            except Exception as e:
                print(f"Error in dense retrieval: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error in dense retrieval: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Invalid retrieval method")
        
        # 检查生成器是否已初始化
        if answer_generator is None:
            raise HTTPException(status_code=500, detail="Answer generator not initialized")
            
        # 生成答案
        try:
            answer = answer_generator.generate(request.question, chunk_texts, generate_type="user")
        except Exception as e:
            print(f"Error in answer generation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in answer generation: {str(e)}")
        
        return {
            "question": request.question,
            "answer": answer,
            "document_id": doc_ids,
            "documents": chunk_texts
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 