from typing import List, Dict, Any
from langchain.schema import Document
from uuid import UUID
import logging

from .base_retriever import BaseCustomRetriever
from ..services.memory_service import memory_service
from ..utils.exceptions import ChainError

logger = logging.getLogger(__name__)

class MemoryRetriever(BaseCustomRetriever):
    """Retriever focused on memory and conversation context"""
    
    def __init__(self, user_id: UUID, session_id: UUID = None):
        super().__init__(user_id, session_id)
    
    async def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents based on memory and conversation context"""
        try:
            documents = []
            
            # Get comprehensive conversation context
            context = await memory_service.get_conversation_context(
                session_id=self.session_id,
                user_id=self.user_id,
                current_query=query
            )
            
            # Convert recent messages to documents
            for msg in context.recent_messages:
                doc = Document(
                    page_content=msg['content'],
                    metadata={
                        'source': 'recent_context',
                        'role': msg['role'],
                        'message_id': msg['message_id'],
                        'token_count': msg.get('token_count', 0),
                        'timestamp': msg['created_at']
                    }
                )
                documents.append(doc)
            
            # Convert relevant history to documents
            for msg_result in context.relevant_history.get('messages', []):
                metadata = msg_result.get('metadata', {})
                doc = Document(
                    page_content=f"Historical context from {metadata.get('session_name', 'previous conversation')}",
                    metadata={
                        'source': 'historical_context',
                        'relevance_score': msg_result.get('score', 0),
                        **metadata
                    }
                )
                documents.append(doc)
            
            # Convert summaries to documents
            for summary in context.summaries:
                doc = Document(
                    page_content=summary['summary_text'],
                    metadata={
                        'source': 'memory_summary',
                        'summary_id': summary['summary_id'],
                        'message_count': summary['message_count'],
                        'timestamp': summary['created_at']
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory documents: {e}")
            raise ChainError(f"Memory retrieval failed: {e}")