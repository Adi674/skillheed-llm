# app/services/memory_service.py - OPTIMIZED WITH BATCH PROCESSING
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
import logging
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..models.enums import MessageRole
from ..models.schemas import RetrievalContext
from .supabase_service import supabase_service
from .vector_service import vector_service
from .embedding_service import embedding_service
from ..utils.exceptions import MemoryServiceError

logger = logging.getLogger(__name__)

class MemoryService:
    """Optimized Memory Service with batch processing"""
    
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
            
            conversation_text = self._format_messages_for_summary(messages_to_summarize)
            summary = await self._generate_summary_with_llm(conversation_text)
            
            start_message_id = UUID(messages_to_summarize[0]['message_id'])
            end_message_id = UUID(messages_to_summarize[-1]['message_id'])
            
            # Use batch embedding for summary
            summary_embedding = await embedding_service.embed_text_batch(summary, str(user_id))
            
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
        """Generate summary using LLM (improved placeholder)"""
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
        """Get conversation context - SIMPLE FIXED VERSION"""
        try:
            max_tokens = max_tokens or self.max_tokens
            
            # Initialize context
            context = RetrievalContext(
                recent_messages=[],
                relevant_history={"messages": [], "summaries": []},
                summaries=[],
                total_tokens=0
            )
            
            # Get query embedding with timeout
            query_embedding = None
            try:
                query_embedding = await asyncio.wait_for(
                    embedding_service.embed_text_batch(current_query, str(user_id)),
                    timeout=1.0
                )
            except Exception as e:
                logger.error(f"Failed to generate query embedding: {e}")
            
            # Get recent messages with timeout
            try:
                context.recent_messages = await asyncio.wait_for(
                    supabase_service.get_recent_messages(session_id, limit=self.context_window),
                    timeout=0.5
                )
            except Exception as e:
                logger.error(f"Failed to get recent messages: {e}")
                context.recent_messages = []
            
            # Get summaries with timeout
            try:
                context.summaries = await asyncio.wait_for(
                    supabase_service.get_session_summaries(session_id, limit=2),
                    timeout=0.5
                )
            except Exception as e:
                logger.error(f"Failed to get session summaries: {e}")
                context.summaries = []
            
            # Get vector search results if embedding successful
            if query_embedding:
                try:
                    vector_results = await asyncio.wait_for(
                        vector_service.hybrid_search(
                            query_embedding=query_embedding,
                            user_id=user_id,
                            session_id=session_id,
                            message_top_k=3,
                            summary_top_k=2
                        ),
                        timeout=1.0
                    )
                    
                    if isinstance(vector_results, dict):
                        context.relevant_history = {
                            "messages": vector_results.get("messages", []),
                            "summaries": vector_results.get("summaries", [])
                        }
                except Exception as e:
                    logger.error(f"Vector search failed: {e}")
                    context.relevant_history = {"messages": [], "summaries": []}
            
            # Calculate tokens
            try:
                context.total_tokens = self._calculate_context_tokens(
                    context.recent_messages, 
                    context.relevant_history, 
                    context.summaries
                )
            except Exception as e:
                logger.error(f"Failed to calculate context tokens: {e}")
                context.total_tokens = 0
            
            logger.info(f"Retrieved context for session {session_id}: "
                    f"{len(context.recent_messages)} recent messages, "
                    f"{len(context.relevant_history.get('messages', []))} historical messages, "
                    f"{len(context.summaries)} summaries, "
                    f"{context.total_tokens} total tokens")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get conversation context: {e}")
            return RetrievalContext(
                recent_messages=[],
                relevant_history={"messages": [], "summaries": []},
                summaries=[],
                total_tokens=0
            )
    
    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get query embedding with error handling"""
        try:
            return await embedding_service.embed_text_batch(query, "context_query")
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None
    
    async def _get_recent_messages(self, session_id: UUID) -> List[Dict[str, Any]]:
        """Get recent messages with error handling"""
        try:
            return await supabase_service.get_recent_messages(session_id, limit=self.context_window)
        except Exception as e:
            logger.error(f"Failed to get recent messages: {e}")
            return []
    
    async def _get_session_summaries(self, session_id: UUID) -> List[Dict[str, Any]]:
        """Get session summaries with error handling"""
        try:
            return await supabase_service.get_session_summaries(session_id, limit=2)
        except Exception as e:
            logger.error(f"Failed to get session summaries: {e}")
            return []
    
    def _calculate_context_tokens(
        self,
        recent_messages: List[Dict[str, Any]],
        vector_results: Dict[str, List[Dict[str, Any]]],
        summaries: List[Dict[str, Any]]
    ) -> int:
        """Calculate total tokens in context with error handling"""
        try:
            total_tokens = 0
            
            for msg in recent_messages:
                total_tokens += msg.get('token_count', 0) or 0
            
            for msg in vector_results.get('messages', []):
                metadata = msg.get('metadata', {})
                total_tokens += metadata.get('token_count', 50) or 50
            
            for summary in summaries:
                summary_text = summary.get('summary_text', '')
                if summary_text:
                    total_tokens += len(summary_text) // 4
            
            return total_tokens
            
        except Exception as e:
            logger.error(f"Token calculation failed: {e}")
            return 0
    
    # OPTIMIZED: Use batch processing for long-term memory storage
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
        """Store message in long-term memory using batch processing"""
        try:
            # Use batch embedding processing
            embedding = await embedding_service.embed_text_batch(content, str(user_id))
            
            # Use batch vector storage
            vector_id = await vector_service.store_message_vector_batch(
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
            
            logger.info(f"Stored message in long-term memory (batch): {vector_id}")
            
        except Exception as e:
            logger.error(f"Failed to store message in long-term memory: {e}")
    
    # OPTIMIZED: Batch processing for multiple messages
    async def store_messages_in_batch(
        self,
        messages: List[Tuple[UUID, UUID, UUID, MessageRole, str, int, datetime]]
    ):
        """Store multiple messages in batch for better performance"""
        try:
            # Extract contents for batch embedding
            contents = [msg[4] for msg in messages]  # msg[4] is content
            user_ids = [str(msg[0]) for msg in messages]  # msg[0] is user_id
            
            # Generate embeddings in batch
            embeddings = await embedding_service.embed_batch_manual(contents)
            
            # Store all vectors in batch
            tasks = []
            for (user_id, session_id, message_id, role, content, token_count, timestamp), embedding in zip(messages, embeddings):
                tasks.append(
                    vector_service.store_message_vector_batch(
                        user_id=user_id,
                        session_id=session_id,
                        message_id=message_id,
                        embedding=embedding,
                        role=role.value,
                        token_count=token_count,
                        timestamp=timestamp
                    )
                )
            
            # Execute in parallel
            vector_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update Pinecone IDs
            update_tasks = []
            for i, result in enumerate(vector_results):
                if not isinstance(result, Exception):
                    update_tasks.append(
                        supabase_service.update_message_pinecone_id(
                            messages[i][2],  # message_id
                            result
                        )
                    )
            
            if update_tasks:
                await asyncio.gather(*update_tasks, return_exceptions=True)
            
            logger.info(f"Batch stored {len(messages)} messages in long-term memory")
            
        except Exception as e:
            logger.error(f"Batch storage failed: {e}")
    
    async def manage_session_memory(
        self,
        session_id: UUID,
        user_id: UUID
    ):
        """Manage session memory by summarizing if needed"""
        try:
            if await self.should_summarize_session(session_id):
                all_messages = await supabase_service.get_session_messages(session_id)
                
                if len(all_messages) > self.context_window:
                    messages_to_summarize = all_messages[:-self.context_window]
                    
                    await self.create_conversation_summary(
                        session_id, user_id, messages_to_summarize
                    )
                    
                    logger.info(f"Session {session_id} memory managed - summary created")
                    
        except Exception as e:
            logger.error(f"Failed to manage session memory: {e}")
    
    async def health_check(self) -> bool:
        """Enhanced health check for memory service"""
        try:
            services_healthy = await asyncio.gather(
                supabase_service.health_check(),
                vector_service.health_check(),
                embedding_service.health_check(),
                return_exceptions=True
            )
            
            healthy_count = sum(1 for result in services_healthy if result is True)
            return healthy_count >= 2
            
        except Exception as e:
            logger.error(f"Memory service health check failed: {e}")
            return False

# Global memory service instance
memory_service = MemoryService()