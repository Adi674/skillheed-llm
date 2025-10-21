from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
import logging
from ..config import settings
from ..models.enums import MessageRole
from ..models.schemas import RetrievalContext
from .supabase_service import supabase_service
from .vector_service import vector_service
from .embedding_service import embedding_service
from ..utils.exceptions import MemoryServiceError

logger = logging.getLogger(__name__)

class MemoryService:
    """Service for managing conversation memory and context"""
    
    def __init__(self):
        self.max_tokens = settings.max_tokens_per_session
        self.context_window = settings.context_window_size
        self.summary_threshold = settings.summary_trigger_threshold
    
    async def should_summarize_session(self, session_id: UUID) -> bool:
        """Check if session needs summarization"""
        try:
            message_count = await supabase_service.count_session_messages(session_id)
            return message_count >= self.summary_threshold
        except Exception as e:
            logger.error(f"Failed to check summarization need: {e}")
            return False
    
    async def create_conversation_summary(
        self,
        session_id: UUID,
        user_id: UUID,
        messages_to_summarize: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Create a summary of conversation messages"""
        try:
            if not messages_to_summarize:
                return None
            
            # Format messages for summarization
            conversation_text = self._format_messages_for_summary(messages_to_summarize)
            
            # For now, create a simple summary
            # In production, you'd use your LLM to generate this
            summary = await self._generate_summary_with_llm(conversation_text)
            
            # Get start and end message IDs
            start_message_id = UUID(messages_to_summarize[0]['message_id'])
            end_message_id = UUID(messages_to_summarize[-1]['message_id'])
            
            # Generate embedding for summary
            summary_embedding = await embedding_service.embed_text(summary)
            
            # Store summary in Supabase
            summary_record = await supabase_service.create_memory_summary(
                session_id=session_id,
                user_id=user_id,
                summary_text=summary,
                message_count=len(messages_to_summarize),
                start_message_id=start_message_id,
                end_message_id=end_message_id,
                embedding=summary_embedding
            )
            
            # Store summary vector in Pinecone
            vector_id = await vector_service.store_summary_vector(
                user_id=user_id,
                session_id=session_id,
                summary_id=summary_record.summary_id,
                embedding=summary_embedding,
                message_count=len(messages_to_summarize),
                start_message_id=start_message_id,
                end_message_id=end_message_id,
                timestamp=datetime.utcnow()
            )
            
            logger.info(f"Created summary for session {session_id}: {vector_id}")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to create conversation summary: {e}")
            raise MemoryServiceError(f"Summary creation failed: {e}")
    
    def _format_messages_for_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages into a readable conversation text"""
        formatted_lines = []
        for msg in messages:
            role = msg['role'].capitalize()
            content = msg['content']
            formatted_lines.append(f"{role}: {content}")
        return "\n".join(formatted_lines)
    
    async def _generate_summary_with_llm(self, conversation_text: str) -> str:
        """Generate summary using LLM (placeholder for now)"""
        # TODO: Implement with Groq LLM
        # For now, return a simple summary
        lines = conversation_text.split('\n')
        user_messages = [line for line in lines if line.startswith('User:')]
        assistant_messages = [line for line in lines if line.startswith('Assistant:')]
        
        summary = f"Conversation with {len(user_messages)} user messages and {len(assistant_messages)} assistant responses. "
        
        if user_messages:
            first_topic = user_messages[0][:100] + "..." if len(user_messages[0]) > 100 else user_messages[0]
            summary += f"Started with: {first_topic}"
        
        return summary
    
    async def get_conversation_context(
        self,
        session_id: UUID,
        user_id: UUID,
        current_query: str,
        max_tokens: Optional[int] = None
    ) -> RetrievalContext:
        """Get comprehensive conversation context for the current query"""
        try:
            max_tokens = max_tokens or self.max_tokens
            
            # Get query embedding
            query_embedding = await embedding_service.embed_text(current_query)
            
            # Get recent messages from current session
            recent_messages = await supabase_service.get_recent_messages(
                session_id, limit=self.context_window
            )
            
            # Get relevant historical context from vectors
            vector_results = await vector_service.hybrid_search(
                query_embedding=query_embedding,
                user_id=user_id,
                session_id=session_id,
                message_top_k=3,
                summary_top_k=2
            )
            
            # Get recent summaries for this session
            session_summaries = await supabase_service.get_session_summaries(
                session_id, limit=2
            )
            
            # Calculate total tokens
            total_tokens = self._calculate_context_tokens(
                recent_messages, vector_results, session_summaries
            )
            
            return RetrievalContext(
                recent_messages=recent_messages,
                relevant_history=vector_results,
                summaries=session_summaries,
                total_tokens=total_tokens
            )
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            raise MemoryServiceError(f"Context retrieval failed: {e}")
    
    def _calculate_context_tokens(
        self,
        recent_messages: List[Dict[str, Any]],
        vector_results: Dict[str, List[Dict[str, Any]]],
        summaries: List[Dict[str, Any]]
    ) -> int:
        """Calculate total tokens in context"""
        total_tokens = 0
        
        # Count tokens in recent messages
        for msg in recent_messages:
            total_tokens += msg.get('token_count', 0)
        
        # Count tokens in vector results (estimate)
        for msg in vector_results.get('messages', []):
            total_tokens += msg.get('metadata', {}).get('token_count', 50)
        
        # Count tokens in summaries (estimate)
        for summary in summaries:
            # Rough estimate: 4 chars per token
            total_tokens += len(summary.get('summary_text', '')) // 4
        
        return total_tokens
    
    async def store_message_in_long_term_memory(
        self,
        user_id: UUID,
        session_id: UUID,
        message_id: UUID,
        role: MessageRole,
        content: str,
        token_count: int,
        timestamp: datetime
    ):
        """Store message in long-term memory (Pinecone)"""
        try:
            # Generate embedding
            embedding = await embedding_service.embed_text(content)
            
            # Store in Pinecone
            vector_id = await vector_service.store_message_vector(
                user_id=user_id,
                session_id=session_id,
                message_id=message_id,
                embedding=embedding,
                role=role.value,
                token_count=token_count,
                timestamp=timestamp
            )
            
            # Update message with Pinecone ID
            await supabase_service.update_message_pinecone_id(message_id, vector_id)
            
            logger.info(f"Stored message in long-term memory: {vector_id}")
            
        except Exception as e:
            logger.error(f"Failed to store message in long-term memory: {e}")
            # Don't raise exception here as it's not critical for chat flow
    
    async def manage_session_memory(
        self,
        session_id: UUID,
        user_id: UUID
    ):
        """Manage session memory by summarizing if needed"""
        try:
            if await self.should_summarize_session(session_id):
                # Get older messages for summarization (excluding recent context window)
                all_messages = await supabase_service.get_session_messages(session_id)
                
                if len(all_messages) > self.context_window:
                    # Messages to summarize (exclude recent ones)
                    messages_to_summarize = all_messages[:-self.context_window]
                    
                    # Create summary
                    await self.create_conversation_summary(
                        session_id, user_id, messages_to_summarize
                    )
                    
                    logger.info(f"Session {session_id} memory managed - summary created")
                    
        except Exception as e:
            logger.error(f"Failed to manage session memory: {e}")
            # Don't raise exception as it's not critical for immediate chat flow
    
    async def health_check(self) -> bool:
        """Check if memory service is healthy"""
        try:
            # Check if dependent services are healthy
            return (
                await supabase_service.health_check() and
                await vector_service.health_check() and
                await embedding_service.health_check()
            )
        except Exception:
            return False

# Global memory service instance
memory_service = MemoryService()