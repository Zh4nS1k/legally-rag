import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import openai
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

@dataclass
class SearchResult:
    """Represents a search result with metadata"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source: str

@dataclass
class ConversationTurn:
    """Represents a conversation turn"""
    user_query: str
    retrieved_context: List[SearchResult]
    generated_response: str
    timestamp: datetime

class EnhancedRAGSystem:
    def __init__(self):
        # Initialize clients
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        index_name = os.getenv("PINECONE_INDEX_NAME")
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME environment variable is required")
        self.index = self.pinecone.Index(index_name)
        
        # Initialize models
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Conversation memory
        self.conversation_history: List[ConversationTurn] = []
        self.max_history_length = 10
        
        # Search parameters
        self.top_k_initial = 20
        self.top_k_final = 5
        self.rerank_threshold = 0.5
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting OpenAI embedding: {e}")
            return self.embedding_model.encode(text).tolist()
    
    def dense_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Perform dense vector search using Pinecone"""
        try:
            query_embedding = self.get_embedding(query)
            
            # Try different Pinecone API formats
            try:
                results = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )  # type: ignore
            except TypeError:
                # Fallback for older Pinecone versions
                results = self.index.query(
                    queries=[query_embedding],
                    top_k=top_k,
                    include_metadata=True
                )  # type: ignore
            
            search_results = []
            try:
                matches = results.matches if hasattr(results, 'matches') else results.get('matches', [])  # type: ignore
                for match in matches:  # type: ignore
                    search_results.append(SearchResult(
                        id=match['id'],
                        text=match['metadata'].get('text', ''),
                        score=match['score'],
                        metadata=match['metadata'],
                        source=match['metadata'].get('filename', 'Unknown')
                    ))
            except (AttributeError, KeyError, TypeError):
                print("Error processing search results")
                return []
            
            return search_results
        except Exception as e:
            print(f"Error in dense search: {e}")
            return []
    
    def sparse_search(self, query: str, documents: List[str]) -> List[float]:
        """Perform sparse search using simple keyword matching"""
        try:
            # Simple keyword-based scoring
            query_words = set(query.lower().split())
            scores = []
            
            for doc in documents:
                doc_words = set(doc.lower().split())
                intersection = query_words.intersection(doc_words)
                score = len(intersection) / len(query_words) if query_words else 0
                scores.append(score)
            
            return scores
        except Exception as e:
            print(f"Error in sparse search: {e}")
            return [0.0] * len(documents)
    
    def hybrid_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Perform hybrid search combining dense and sparse retrieval"""
        # Dense search
        dense_results = self.dense_search(query, top_k)
        
        if not dense_results:
            return []
        
        # Extract texts for sparse search
        texts = [result.text for result in dense_results]
        
        # Sparse search
        sparse_scores = self.sparse_search(query, texts)
        
        # Combine scores (weighted average)
        alpha = 0.7  # Weight for dense search
        combined_results = []
        
        for i, result in enumerate(dense_results):
            combined_score = alpha * result.score + (1 - alpha) * sparse_scores[i]
            combined_results.append(SearchResult(
                id=result.id,
                text=result.text,
                score=combined_score,
                metadata=result.metadata,
                source=result.source
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return combined_results[:top_k]
    
    def rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank results using cross-encoder"""
        if not results:
            return results
        
        try:
            # Prepare pairs for cross-encoder
            pairs = [(query, result.text) for result in results]
            
            # Get cross-encoder scores
            scores = self.cross_encoder.predict(pairs)
            
            # Update scores and sort
            for i, result in enumerate(results):
                result.score = scores[i]
            
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Filter by threshold
            filtered_results = [r for r in results if r.score > self.rerank_threshold]
            
            return filtered_results[:self.top_k_final]
        except Exception as e:
            print(f"Error in re-ranking: {e}")
            return results[:self.top_k_final]
    
    def build_context(self, results: List[SearchResult], max_tokens: int = 4000) -> str:
        """Build context from search results with token limit"""
        context_parts = []
        current_tokens = 0
        
        for result in results:
            # Simple token estimation (rough approximation)
            result_tokens = len(result.text.split()) * 1.3
            
            if current_tokens + result_tokens > max_tokens:
                break
            
            context_parts.append(f"[Source: {result.source}]\n{result.text}")
            current_tokens += result_tokens
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, query: str, context: str, conversation_history: Optional[List[ConversationTurn]] = None) -> str:
        """Generate response using OpenAI with conversation history"""
        try:
            # Build system prompt
            system_prompt = """Ты — AI-юрист, специализирующийся на казахстанском законодательстве. 
            Твоя задача — давать точные, полезные ответы на основе предоставленного контекста.
            
            Правила:
            1. Отвечай только на основе предоставленного контекста
            2. Если информации недостаточно, честно скажи об этом
            3. Цитируй конкретные статьи и положения
            4. Объясняй сложные юридические концепции простым языком
            5. Всегда указывай источник информации"""
            
            # Build conversation context
            messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history if provided
            if conversation_history:
                for turn in conversation_history[-3:]:  # Last 3 turns
                    messages.append({"role": "user", "content": turn.user_query})
                    messages.append({"role": "assistant", "content": turn.generated_response})
            
            # Add current query with context
            user_message = f"Контекст:\n{context}\n\nВопрос: {query}"
            messages.append({"role": "user", "content": user_message})
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,  # type: ignore
                temperature=0.3,
                max_tokens=1000
            )
            
            response_content = response.choices[0].message.content
            return response_content if response_content else "Извините, не удалось сгенерировать ответ."
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Извините, произошла ошибка при генерации ответа."
    
    def query(self, user_query: str, use_hybrid_search: bool = True, use_reranking: bool = True) -> Dict[str, Any]:
        """Main query method"""
        try:
            # Perform search
            if use_hybrid_search:
                search_results = self.hybrid_search(user_query, self.top_k_initial)
            else:
                search_results = self.dense_search(user_query, self.top_k_initial)
            
            if not search_results:
                return {
                    "answer": "К сожалению, не удалось найти релевантную информацию по вашему запросу.",
                    "sources": [],
                    "search_results": []
                }
            
            # Re-rank if enabled
            if use_reranking:
                search_results = self.rerank_results(user_query, search_results)
            
            # Build context
            context = self.build_context(search_results)
            
            # Generate response
            response = self.generate_response(user_query, context, self.conversation_history)
            
            # Update conversation history
            conversation_turn = ConversationTurn(
                user_query=user_query,
                retrieved_context=search_results,
                generated_response=response,
                timestamp=datetime.now()
            )
            
            self.conversation_history.append(conversation_turn)
            
            # Keep history within limit
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            return {
                "answer": response,
                "sources": [result.source for result in search_results],
                "search_results": [
                    {
                        "id": result.id,
                        "text": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                        "score": result.score,
                        "source": result.source
                    }
                    for result in search_results
                ],
                "context_length": len(context),
                "results_count": len(search_results)
            }
            
        except Exception as e:
            print(f"Error in query: {e}")
            return {
                "answer": "Произошла ошибка при обработке запроса.",
                "sources": [],
                "search_results": []
            }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return [
            {
                "user_query": turn.user_query,
                "response": turn.generated_response,
                "timestamp": turn.timestamp.isoformat(),
                "sources": [r.source for r in turn.retrieved_context]
            }
            for turn in self.conversation_history
        ]
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            index_stats = self.index.describe_index_stats()
            return {
                "total_vectors": index_stats.get("total_vector_count", 0),
                "index_dimension": index_stats.get("dimension", 0),
                "conversation_history_length": len(self.conversation_history),
                "models": {
                    "embedding": "text-embedding-3-small",
                    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    "generation": "gpt-4"
                }
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"error": str(e)}

# Global RAG instance
rag_system = EnhancedRAGSystem() 