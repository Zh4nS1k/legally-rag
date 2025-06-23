from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from rag_system import rag_system

# === Инициализация ===
app = FastAPI(
    title="Enhanced Legal RAG System",
    description="AI-powered legal document search and analysis system",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# === Модели запросов ===
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    use_hybrid_search: Optional[bool] = True
    use_reranking: Optional[bool] = True

class ConversationRequest(BaseModel):
    user_id: str

class StatsRequest(BaseModel):
    user_id: Optional[str] = None

# === Эндпоинты ===
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Enhanced Legal RAG System",
        "version": "2.0.0",
        "features": [
            "Hybrid search (dense + sparse)",
            "Cross-encoder re-ranking",
            "Conversation memory",
            "Semantic chunking",
            "Enhanced context building"
        ]
    }

@app.post("/search")
async def search_docs(req: QueryRequest):
    """Enhanced search endpoint with hybrid search and re-ranking"""
    try:
        # Use the enhanced RAG system
        result = rag_system.query(
            user_query=req.query,
            use_hybrid_search=req.use_hybrid_search if req.use_hybrid_search is not None else True,
            use_reranking=req.use_reranking if req.use_reranking is not None else True
        )
        
        return {
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "search_results": result["search_results"],
            "metadata": {
                "context_length": result["context_length"],
                "results_count": result["results_count"],
                "search_type": "hybrid" if (req.use_hybrid_search if req.use_hybrid_search is not None else True) else "dense",
                "reranking": req.use_reranking if req.use_reranking is not None else True
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{user_id}")
async def get_conversation_history(user_id: str):
    """Get conversation history for a user"""
    try:
        # For now, return global conversation history
        # In a real system, you'd filter by user_id
        history = rag_system.get_conversation_history()
        return {
            "success": True,
            "conversation_history": history,
            "user_id": user_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversation/{user_id}")
async def clear_conversation_history(user_id: str):
    """Clear conversation history for a user"""
    try:
        rag_system.clear_conversation_history()
        return {
            "success": True,
            "message": f"Conversation history cleared for user {user_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    try:
        stats = rag_system.get_system_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/simple")
async def simple_search(req: QueryRequest):
    """Simple search without hybrid search or re-ranking"""
    try:
        result = rag_system.query(
            user_query=req.query,
            use_hybrid_search=False,
            use_reranking=False
        )
        
        return {
            "success": True,
            "answer": result["answer"],
            "sources": result["sources"],
            "search_results": result["search_results"],
            "metadata": {
                "search_type": "simple",
                "context_length": result["context_length"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test basic functionality
        test_result = rag_system.query("test", use_hybrid_search=False, use_reranking=False)
        return {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "version": "2.0.0"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
