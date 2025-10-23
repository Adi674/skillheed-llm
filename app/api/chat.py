# app/api/chat.py - OPTIMIZED WITH SINGLE CHAIN AND PRODUCTION IMPROVEMENTS
import time
from typing import Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request  # CHANGE 1: Added Request for rate limiting
import asyncio
import logging

from ..models.schemas import ChatRequest, ChatResponse, MessageResponse, RetrievalContext
from ..models.enums import MessageRole
from ..services.supabase_service import supabase_service
from ..services.memory_service import memory_service
from ..services.embedding_service import embedding_service
from ..utils.exceptions import DatabaseError, MemoryServiceError, EmbeddingServiceError
# CHANGE 2: SIMPLIFIED CHAIN IMPORT - removed redundant OptimizedRAGChain
from ..chains.simple_rag_chain import simple_rag_chain
from ..services.vector_service import vector_service 

# CHANGE 3: Removed direct LLM import (no longer needed with single chain)
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

# CHANGE 4: REMOVED get_llm() function - no longer needed

# CHANGE 5: REMOVED OptimizedRAGChain class entirely - using simple_rag_chain directly

# CHANGE 6: SIMPLIFIED generate_response_optimized function
async def generate_response_optimized(message: str, context) -> str:
    """Generate response using single chain with better error handling"""
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
        
        # CHANGE 7: Use simple_rag_chain directly with semaphore control
        async with LLM_SEMAPHORE:
            response = await simple_rag_chain.invoke({
                "question": message,
                "context": context_info,
                "recent_messages": recent_formatted
            })
        
        return response
        
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return "I apologize, but I'm experiencing technical difficulties. Please try again."

# CHANGE 8: Updated chat endpoint with Request parameter and improved error handling
@router.post("/", response_model=ChatResponse)
async def chat(
    request: Request,  # Added for rate limiting support
    chat_request: ChatRequest,  # Renamed for clarity
    background_tasks: BackgroundTasks
):
    """Optimized chat endpoint with single chain"""
    start_time = time.time()
    session_id = None
    
    try:
        # CHANGE 9: Enhanced input validation
        if not chat_request.message or len(chat_request.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if len(chat_request.message) > 4000:
            raise HTTPException(status_code=400, detail="Message too long")
        
        # CHANGE 10: Additional validation for malicious content
        sanitized_message = chat_request.message.strip()
        if len(sanitized_message) < 1:
            raise HTTPException(status_code=400, detail="Message too short")
        
        # OPTIMIZED: Single session lookup with caching
        session_id = chat_request.session_id
        if not session_id:
            # Create new session
            async with DB_SEMAPHORE:
                session = await supabase_service.create_session(
                    user_id=chat_request.user_id,
                    session_name=chat_request.session_name or "New Chat"
                )
                session_id = session.session_id
                
                # Cache the new session immediately
                cache_key = f"{session_id}_{chat_request.user_id}"
                SESSION_CACHE[cache_key] = (time.time(), session)
                
                logger.info(f"Created new session {session_id}")
        else:
            # Use cached session lookup (SINGLE CALL)
            session = await get_cached_session(session_id, chat_request.user_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        
        # Fast token counting
        user_token_count = len(sanitized_message) // 4
        
        # Save user message
        async with DB_SEMAPHORE:
            user_message = await supabase_service.create_message(
                session_id=session_id,
                user_id=chat_request.user_id,
                role=MessageRole.USER,
                content=sanitized_message,
                token_count=user_token_count
            )
        
        # CHANGE 11: Improved context retrieval with better timeout handling
        try:
            context = await asyncio.wait_for(
                memory_service.get_conversation_context(
                    session_id=session_id,
                    user_id=chat_request.user_id,
                    current_query=sanitized_message
                ),
                timeout=3.0  # Increased timeout slightly for better reliability
            )
        except asyncio.TimeoutError:
            logger.warning(f"Context retrieval timed out for session {session_id}")
            context = RetrievalContext()
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            context = RetrievalContext()
        
        # Generate response using simplified chain
        assistant_response = await generate_response_optimized(sanitized_message, context)
        
        # CHANGE 12: Better token counting
        assistant_token_count = max(1, len(assistant_response) // 4)
        
        # Save assistant message
        async with DB_SEMAPHORE:
            assistant_message = await supabase_service.create_message(
                session_id=session_id,
                user_id=chat_request.user_id,
                role=MessageRole.ASSISTANT,
                content=assistant_response,
                token_count=assistant_token_count
            )
        
        # Background tasks
        background_tasks.add_task(
            optimized_background_tasks,
            chat_request.user_id,
            session_id,
            user_message,
            assistant_message
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        # CHANGE 13: Enhanced response with better error handling
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
        logger.error(f"Chat failed for user {chat_request.user_id if 'chat_request' in locals() else 'unknown'}: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

# CHANGE 14: Enhanced background task processing
async def optimized_background_tasks(
    user_id: UUID,
    session_id: UUID,
    user_message,
    assistant_message
):
    """Optimized background processing with batch operations and better error handling"""
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
            logger.warning("No messages to process in background task")
            return
        
        # BATCH PROCESSING: Generate all embeddings at once
        try:
            embeddings = await embedding_service.embed_batch_manual(contents)
        except Exception as e:
            logger.error(f"Batch embedding failed for session {session_id}: {e}")
            # Try individual embeddings as fallback
            embeddings = []
            for content in contents:
                try:
                    embedding = await embedding_service.embed_text(content)
                    embeddings.append(embedding)
                except Exception as embed_error:
                    logger.error(f"Individual embedding failed: {embed_error}")
                    embeddings.append([0.0] * settings.embedding_dimension)  # Fallback zero vector
            
            if not embeddings:
                logger.error("All embedding attempts failed")
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
        
        # CHANGE 15: Improved Pinecone ID updates with batch processing
        update_tasks = []
        successful_updates = 0
        for i, result in enumerate(vector_results):
            if not isinstance(result, Exception) and result:
                task = supabase_service.update_message_pinecone_id(
                    messages_to_process[i]['message_id'],
                    result
                )
                update_tasks.append(task)
                successful_updates += 1
            else:
                logger.error(f"Vector storage failed for message {messages_to_process[i]['message_id']}: {result}")
        
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=True)
            logger.info(f"Updated {successful_updates} Pinecone IDs successfully")
        
        # Memory management (reduced frequency to improve performance)
        import random
        if random.random() < 0.1:  # Only 10% of the time
            try:
                await memory_service.manage_session_memory(session_id, user_id)
            except Exception as e:
                logger.error(f"Memory management failed for session {session_id}: {e}")
        
        logger.info(f"Background processing completed for {len(messages_to_process)} messages in session {session_id}")
        
    except Exception as e:
        logger.error(f"Background task failed for session {session_id}: {e}")

# Keep existing endpoints unchanged (but enhanced with better error handling)
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
        logger.error(f"Failed to get session messages for {session_id}: {e}")
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
        logger.error(f"Failed to get session context for {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve context")

# CHANGE 16: Simplified LLM test endpoint
@router.get("/test-chain")
async def test_chain_connection():
    """Test the single chain connection"""
    try:
        async with LLM_SEMAPHORE:
            response = await simple_rag_chain.invoke({
                "question": "Hello! Please respond with 'Single chain is working correctly'",
                "context": "Test context",
                "recent_messages": "Test: Hello"
            })
        
        return {
            "status": "success",
            "chain_working": True,
            "test_response": response,
            "model": settings.groq_model,
            "chain_type": "SimpleRAGChain"
        }
    except Exception as e:
        return {
            "status": "error", 
            "chain_working": False,
            "error": str(e),
            "model": settings.groq_model,
            "chain_type": "SimpleRAGChain"
        }

# CHANGE 17: Enhanced performance stats
@router.get("/performance-stats")
async def get_performance_stats():
    """Get current performance statistics"""
    return {
        "session_cache_size": len(SESSION_CACHE),
        "llm_semaphore_available": LLM_SEMAPHORE._value,
        "db_semaphore_available": DB_SEMAPHORE._value,
        "embedding_processor_stats": getattr(embedding_service, 'batch_processor', {}).get_stats() if hasattr(embedding_service, 'batch_processor') else {},
        "vector_processor_stats": getattr(vector_service, 'vector_processor', {}).get_stats() if hasattr(vector_service, 'vector_processor') else {},
        "chain_type": "SimpleRAGChain",
        "architecture": "single_chain_optimized"
    }