import time
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, BackgroundTasks
from ..models.schemas import ChatRequest, ChatResponse, MessageResponse
from ..models.enums import MessageRole
from ..services.supabase_service import supabase_service
from ..services.memory_service import memory_service
from ..services.embedding_service import embedding_service
from ..utils.exceptions import DatabaseError, MemoryServiceError, EmbeddingServiceError

router = APIRouter()

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks
):
    """Main chat endpoint"""
    start_time = time.time()
    
    try:
        # Get or create session
        session_id = request.session_id
        if not session_id:
            # Create new session
            session = await supabase_service.create_session(
                user_id=request.user_id,
                session_name=request.session_name or "New Chat"
            )
            session_id = session.session_id
        else:
            # Verify session exists and belongs to user
            session = await supabase_service.get_session(session_id, request.user_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        
        # Count tokens in user message
        user_token_count = embedding_service.count_tokens(request.message)
        
        # Save user message
        user_message = await supabase_service.create_message(
            session_id=session_id,
            user_id=request.user_id,
            role=MessageRole.USER,
            content=request.message,
            token_count=user_token_count
        )
        
        # Get conversation context
        context = await memory_service.get_conversation_context(
            session_id=session_id,
            user_id=request.user_id,
            current_query=request.message
        )
        
        # Generate response using LangChain (placeholder for now)
        assistant_response = await generate_response(request.message, context)
        
        # Count tokens in assistant response
        assistant_token_count = embedding_service.count_tokens(assistant_response)
        
        # Save assistant message
        assistant_message = await supabase_service.create_message(
            session_id=session_id,
            user_id=request.user_id,
            role=MessageRole.ASSISTANT,
            content=assistant_response,
            token_count=assistant_token_count
        )
        
        # Background tasks for memory management
        background_tasks.add_task(
            background_memory_tasks,
            request.user_id,
            session_id,
            user_message,
            assistant_message
        )
        
        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        
        return ChatResponse(
            message=assistant_response,
            session_id=session_id,
            message_id=assistant_message.message_id,
            token_count=assistant_token_count,
            response_time_ms=response_time_ms,
            context_used=context.total_tokens
        )
        
    except HTTPException:
        raise
    except (DatabaseError, MemoryServiceError, EmbeddingServiceError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to process chat message")

async def generate_response(message: str, context) -> str:
    """Generate response using LangChain (placeholder)"""
    # TODO: Implement with LangChain RAG chain
    # For now, return a simple response
    
    # Format context for response
    context_info = ""
    if context.recent_messages:
        context_info += f"Recent conversation with {len(context.recent_messages)} messages. "
    if context.relevant_history.get('messages'):
        context_info += f"Found {len(context.relevant_history['messages'])} relevant historical messages. "
    if context.relevant_history.get('summaries'):
        context_info += f"Found {len(context.relevant_history['summaries'])} relevant conversation summaries. "
    
    return f"I understand your message: '{message}'. {context_info}This is a placeholder response. In production, this would use the RAG chain with Groq LLM."

async def background_memory_tasks(
    user_id: UUID,
    session_id: UUID, 
    user_message: MessageResponse,
    assistant_message: MessageResponse
):
    """Background tasks for memory management"""
    try:
        # Store messages in long-term memory if needed
        await memory_service.store_message_in_long_term_memory(
            user_id=user_id,
            session_id=session_id,
            message_id=user_message.message_id,
            role=MessageRole.USER,
            content=user_message.content,
            token_count=user_message.token_count or 0,
            timestamp=user_message.created_at
        )
        
        await memory_service.store_message_in_long_term_memory(
            user_id=user_id,
            session_id=session_id,
            message_id=assistant_message.message_id,
            role=MessageRole.ASSISTANT,
            content=assistant_message.content,
            token_count=assistant_message.token_count or 0,
            timestamp=assistant_message.created_at
        )
        
        # Manage session memory (summarization if needed)
        await memory_service.manage_session_memory(session_id, user_id)
        
    except Exception as e:
        # Log error but don't fail the request
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Background memory task failed: {e}")

@router.get("/{session_id}/messages")
async def get_session_messages(
    session_id: UUID,
    user_id: UUID,
    limit: Optional[int] = 50
):
    """Get messages for a session"""
    try:
        # Verify session belongs to user
        session = await supabase_service.get_session(session_id, user_id)
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
        raise HTTPException(status_code=500, detail="Failed to retrieve messages")

@router.get("/{session_id}/context")
async def get_session_context(
    session_id: UUID,
    user_id: UUID,
    query: str
):
    """Get conversation context for debugging"""
    try:
        # Verify session belongs to user
        session = await supabase_service.get_session(session_id, user_id)
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
        raise HTTPException(status_code=500, detail="Failed to retrieve context")