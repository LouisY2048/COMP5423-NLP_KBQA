"""
Main API server for the Knowledge Base Question Answering System.
This module provides the FastAPI application that handles:
1. Document retrieval using dense retrieval methods
2. Answer generation using a large language model
3. API endpoints for question answering and health checks

Usage:
    python main.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from contextlib import asynccontextmanager

from retrieval.dense_retrieval import DenseRetriever
from generation.answer_generator import AnswerGenerator

dense_retriever = None
answer_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global dense_retriever, answer_generator
    print("Loading retrieval models...")
    dense_retriever = DenseRetriever()
    print("Loading answer generation model...")
    answer_generator = AnswerGenerator()
    print("System ready!")
    
    yield
    
    print("Shutting down...")

app = FastAPI(title="KBQA System API", lifespan=lifespan)

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
    """Ask a question to the KBQA system and get an answer"""
    try:
        if dense_retriever is None:
            raise HTTPException(status_code=500, detail="Dense retriever not initialized")
            
        if request.retrieval_method == "dense":
            try:
                result = dense_retriever.retrieve(request.question)
                doc_ids = result["document_id"]
                
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
        
        if answer_generator is None:
            raise HTTPException(status_code=500, detail="Answer generator not initialized")
            
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