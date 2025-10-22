# app/api/chat.py - OPTIMIZED WITH BATCH PROCESSING
import time
from typing import Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks
import asyncio
import logging

from ..models.schemas import ChatRequest, ChatResponse, MessageResponse, RetrievalContext
from ..models.enums import MessageRole
from ..services.supabase_service import supabase_service
from ..services.memory_service import memory_service
from ..services.embedding_service import embedding_service
from ..utils.exceptions import DatabaseError, MemoryServiceError, EmbeddingServiceError
from ..chains.simple_rag_chain import simple_rag_chain
from ..services.vector_service import vector_service 

# Direct LLM import
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from ..config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# SEMAPHORES FOR RESOURCE MANAGEMENT
LLM_SEMAPHORE = asyncio.Semaphore(25)  # Max 25 concurrent LLM calls (Groq rate limit)
DB_SEMAPHORE = asyncio.Semaphore(50)   # Max 50 concurrent DB operations (Supabase limit)

# SESSION CACHE FOR PERFORMANCE
SESSION_CACHE = {}
CACHE_TTL = 300  # 5 minutes

async def get_cached_session(session_id: UUID, user_id: UUID):
    """Fast session caching to reduce database calls"""
    cache_key = f"{session_id}_{user_id}"
    current_time = time.time()
    
    # Check cache first
    if cache_key in SESSION_CACHE:
        cached_time, session = SESSION_CACHE[cache_key]
        if current_time - cached_time < CACHE_TTL:
            return session
    
    # Fetch with rate limiting
    async with DB_SEMAPHORE:
        session = await supabase_service.get_session(session_id, user_id)
    
    # Cache result
    SESSION_CACHE[cache_key] = (current_time, session)
    
    # Cleanup old entries periodically
    if len(SESSION_CACHE) > 200:
        old_keys = [
            key for key, (cached_time, _) in SESSION_CACHE.items()
            if current_time - cached_time > CACHE_TTL
        ]
        for key in old_keys[:100]:  # Remove oldest 100 entries
            del SESSION_CACHE[key]
    
    return session

def get_llm():
    """Get LLM instance"""
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
        streaming=settings.groq_streaming
    )

class OptimizedRAGChain:
    """Optimized RAG chain with rate limiting"""
    
    def __init__(self):
        self.llm = get_llm()
        self._response_cache = {}
        self._cache_ttl = 300
    
    async def invoke(self, inputs: dict[str, any]) -> str:
        """Process with LLM rate limiting and caching"""
        try:
            # Use semaphore to limit concurrent LLM calls
            async with LLM_SEMAPHORE:
                # Extract inputs
                question = inputs.get("question", "")
                context = inputs.get("context", "")
                recent_messages = inputs.get("recent_messages", "")
                
                # Format prompt
                formatted_prompt = self._format_prompt(question, context, recent_messages)
                
                # Add timeout to prevent hanging requests
                response = await asyncio.wait_for(
                    self.llm.ainvoke([HumanMessage(content=formatted_prompt)]),
                    timeout=30.0  # 30 second timeout
                )
                
                if hasattr(response, 'content'):
                    return response.content.strip()
                else:
                    return str(response).strip()
                    
        except asyncio.TimeoutError:
            logger.error("LLM call timed out")
            return "I apologize for the delay. The system is under high load. Please try again."
        except Exception as e:
            logger.error(f"Optimized chain processing failed: {e}")
            return f"I'm having trouble processing your request. Please try again."
    
    def _format_prompt(self, question: str, context: str = "", recent_messages: str = "") -> str:
        """Format prompt for LLM"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. Use the provided context to give accurate and relevant responses.

Context:
{context if context else "No additional context available."}

Recent conversation:
{recent_messages if recent_messages else "No recent conversation history."}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# Global optimized chain
optimized_rag_chain = OptimizedRAGChain()

async def generate_response_optimized(message: str, context) -> str:
    """Generate response using optimized chain with better error handling"""
    try:
        # Format context efficiently
        context_info = ""
        if hasattr(context, 'relevant_history') and isinstance(context.relevant_history, dict):
            messages = context.relevant_history.get('messages', [])
            summaries = context.relevant_history.get('summaries', [])
            if messages:
                context_info += f"Found {len(messages)} relevant past messages. "
            if summaries:
                context_info += f"Found {len(summaries)} conversation summaries. "
        
        # Format recent messages (limit to last 3 for efficiency)
        recent_formatted = ""
        if hasattr(context, 'recent_messages') and context.recent_messages:
            recent_formatted = "\n".join([
                f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')[:100]}" 
                for msg in context.recent_messages[-3:]  # Only last 3 messages
                if msg.get('content')
            ])
        
        # Use optimized chain
        response = await optimized_rag_chain.invoke({
            "question": message,
            "context": context_info,
            "recent_messages": recent_formatted
        })
        
        return response
        
    except Exception as e:
        logger.error(f"Optimized response generation failed: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again."

# In chat.py, update the chat endpoint to use caching properly:

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """Optimized chat endpoint"""
    start_time = time.time()
    session_id = None
    
    try:
        # Input validation
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(request.message) > 4000:
            raise HTTPException(status_code=400, detail="Message too long")
        
        sanitized_message = request.message.strip()
        
        # OPTIMIZED: Single session lookup with caching
        session_id = request.session_id
        if not session_id:
            # Create new session
            async with DB_SEMAPHORE:
                session = await supabase_service.create_session(
                    user_id=request.user_id,
                    session_name=request.session_name or "New Chat"
                )
                session_id = session.session_id
                
                # Cache the new session immediately
                cache_key = f"{session_id}_{request.user_id}"
                SESSION_CACHE[cache_key] = (time.time(), session)
                
                logger.info(f"Created new session {session_id}")
        else:
            # Use cached session lookup (SINGLE CALL)
            session = await get_cached_session(session_id, request.user_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        
        # Fast token counting
        user_token_count = len(sanitized_message) // 4
        
        # Save user message
        async with DB_SEMAPHORE:
            user_message = await supabase_service.create_message(
                session_id=session_id,
                user_id=request.user_id,
                role=MessageRole.USER,
                content=sanitized_message,
                token_count=user_token_count
            )
        
        # Get context with shorter timeout
        try:
            context = await asyncio.wait_for(
                memory_service.get_conversation_context(
                    session_id=session_id,
                    user_id=request.user_id,
                    current_query=sanitized_message
                ),
                timeout=2.0  # Reduced to 2 seconds
            )
        except asyncio.TimeoutError:
            logger.warning("Context retrieval timed out")
            context = RetrievalContext()
        
        # Generate response
        assistant_response = await generate_response_optimized(sanitized_message, context)
        
        assistant_token_count = len(assistant_response) // 4
        
        # Save assistant message (reuse session from cache, no DB call needed)
        async with DB_SEMAPHORE:
            assistant_message = await supabase_service.create_message(
                session_id=session_id,
                user_id=request.user_id,
                role=MessageRole.ASSISTANT,
                content=assistant_response,
                token_count=assistant_token_count
            )
        
        # Background tasks
        background_tasks.add_task(
            optimized_background_tasks,
            request.user_id,
            session_id,
            user_message,
            assistant_message
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        return ChatResponse(
            message=assistant_response,
            session_id=session_id,
            message_id=assistant_message.message_id,
            token_count=assistant_token_count,
            response_time_ms=response_time_ms,
            context_used=getattr(context, 'total_tokens', 0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

async def optimized_background_tasks(
    user_id: UUID,
    session_id: UUID,
    user_message,
    assistant_message
):
    """Optimized background processing with batch operations"""
    try:
        # Collect messages for batch processing
        messages_to_process = []
        contents = []
        
        if hasattr(user_message, 'message_id'):
            messages_to_process.append({
                'user_id': user_id,
                'session_id': session_id,
                'message_id': user_message.message_id,
                'role': MessageRole.USER,
                'content': user_message.content,
                'token_count': getattr(user_message, 'token_count', 0),
                'timestamp': getattr(user_message, 'created_at', datetime.utcnow())
            })
            contents.append(user_message.content)
        
        if hasattr(assistant_message, 'message_id'):
            messages_to_process.append({
                'user_id': user_id,
                'session_id': session_id,
                'message_id': assistant_message.message_id,
                'role': MessageRole.ASSISTANT,
                'content': assistant_message.content,
                'token_count': getattr(assistant_message, 'token_count', 0),
                'timestamp': getattr(assistant_message, 'created_at', datetime.utcnow())
            })
            contents.append(assistant_message.content)
        
        if not messages_to_process:
            return
        
        # BATCH PROCESSING: Generate all embeddings at once
        try:
            embeddings = await embedding_service.embed_batch_manual(contents)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return
        
        # PARALLEL PROCESSING: Store all vectors simultaneously
        vector_tasks = []
        for msg, embedding in zip(messages_to_process, embeddings):
            task = vector_service.store_message_vector_batch(
                user_id=msg['user_id'],
                session_id=msg['session_id'],
                message_id=msg['message_id'],
                embedding=embedding,
                role=msg['role'].value,
                token_count=msg['token_count'],
                timestamp=msg['timestamp']
            )
            vector_tasks.append(task)
        
        # Execute all vector operations in parallel
        vector_results = await asyncio.gather(*vector_tasks, return_exceptions=True)
        
        # PARALLEL PROCESSING: Update all Pinecone IDs simultaneously
        update_tasks = []
        for i, result in enumerate(vector_results):
            if not isinstance(result, Exception):
                task = supabase_service.update_message_pinecone_id(
                    messages_to_process[i]['message_id'],
                    result
                )
                update_tasks.append(task)
        
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)
        
        # Memory management (reduced frequency to improve performance)
        import random
        if random.random() < 0.1:  # Only 10% of the time
            try:
                await memory_service.manage_session_memory(session_id, user_id)
            except Exception as e:
                logger.error(f"Memory management failed: {e}")
        
        logger.info(f"Optimized background processing completed for {len(messages_to_process)} messages")
        
    except Exception as e:
        logger.error(f"Optimized background task failed: {e}")

# Keep existing endpoints unchanged
@router.get("/{session_id}/messages")
async def get_session_messages(
    session_id: UUID,
    user_id: UUID,
    limit: Optional[int] = 50
):
    """Get messages for a session"""
    try:
        session = await get_cached_session(session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = await supabase_service.get_session_messages(
            session_id=session_id,
            limit=limit
        )
        
        return {"messages": messages, "total": len(messages)}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve messages")

@router.get("/{session_id}/context")
async def get_session_context(
    session_id: UUID,
    user_id: UUID,
    query: str
):
    """Get conversation context for debugging"""
    try:
        session = await get_cached_session(session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        context = await memory_service.get_conversation_context(
            session_id=session_id,
            user_id=user_id,
            current_query=query
        )
        
        return context
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session context: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve context")

@router.get("/test-llm")
async def test_llm_connection():
    """Test LLM connection"""
    try:
        async with LLM_SEMAPHORE:
            llm = get_llm()
            test_response = await llm.ainvoke([
                HumanMessage(content="Hello! Please respond with 'LLM is working correctly'")
            ])
        
        content = test_response.content if hasattr(test_response, 'content') else str(test_response)
        
        return {
            "status": "success",
            "llm_working": True,
            "test_response": content,
            "model": settings.groq_model
        }
    except Exception as e:
        return {
            "status": "error", 
            "llm_working": False,
            "error": str(e),
            "model": settings.groq_model
        }

@router.get("/performance-stats")
async def get_performance_stats():
    """Get current performance statistics"""
    return {
        "session_cache_size": len(SESSION_CACHE),
        "llm_semaphore_available": LLM_SEMAPHORE._value,
        "db_semaphore_available": DB_SEMAPHORE._value,
        "embedding_processor_stats": getattr(embedding_service, 'batch_processor', {}).get_stats() if hasattr(embedding_service, 'batch_processor') else {},
        "vector_processor_stats": getattr(vector_service, 'vector_processor', {}).get_stats() if hasattr(vector_service, 'vector_processor') else {}
    }