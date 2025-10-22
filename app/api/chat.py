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

# Direct LLM import to avoid chain import issues
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from ..config import settings

router = APIRouter()

# Initialize LLM directly
def get_llm():
    """Get LLM instance directly"""
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=settings.groq_temperature,
        max_tokens=settings.groq_max_tokens,
        streaming=settings.groq_streaming
    )

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
        
        # Generate response using LLM directly (no chain import issues)
        assistant_response = await generate_response_direct(request.message, context)
        
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
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")

async def generate_response_direct(message: str, context) -> str:
    """Generate response using Groq LLM directly - no chain imports"""
    try:
        # Format context for response
        context_info = ""
        if context.recent_messages:
            recent_formatted = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in context.recent_messages[-5:]  # Last 5 messages
            ])
            context_info += f"Recent conversation:\n{recent_formatted}\n\n"
        
        if context.relevant_history.get('messages'):
            context_info += f"Found {len(context.relevant_history['messages'])} relevant historical messages. "
        
        if context.relevant_history.get('summaries'):
            context_info += f"Found {len(context.relevant_history['summaries'])} relevant conversation summaries. "
        
        # Create prompt for Llama 3.3
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. Provide accurate, relevant, and concise responses based on the conversation context and retrieved information.

{context_info}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{message}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Get LLM and call it
        llm = get_llm()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        
        # Extract content
        if hasattr(response, 'content'):
            return response.content.strip()
        else:
            return str(response).strip()
            
    except Exception as e:
        return f"I apologize, but I'm having trouble connecting to the AI service. Error: {str(e)}. Please try again."

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

@router.get("/test-llm")
async def test_llm_connection():
    """Test endpoint to verify LLM is working"""
    try:
        llm = get_llm()
        test_response = await llm.ainvoke([
            HumanMessage(content="Hello! Please respond with 'LLM is working correctly' to confirm the connection.")
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