from typing import List, Dict, Any
from langchain.schema import Document
from uuid import UUID
import logging

from .base_retriever import BaseCustomRetriever
from ..services.supabase_service import supabase_service
from ..services.vector_service import vector_service
from ..services.embedding_service import embedding_service
from ..utils.exceptions import ChainError

logger = logging.getLogger(__name__)

class HybridRetriever(BaseCustomRetriever):
    """Custom retriever that combines Supabase and Pinecone results"""
    
    def __init__(self, user_id: UUID, session_id: UUID = None, top_k: int = 5):
        super().__init__(user_id, session_id)
        self.top_k = top_k
    
    async def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve relevant documents from both Supabase and Pinecone"""
        try:
            documents = []
            
            # Get query embedding
            query_embedding = await embedding_service.embed_text(query)
            
            # Get recent messages from current session
            if self.session_id:
                recent_messages = await supabase_service.get_recent_messages(
                    session_id=self.session_id,
                    limit=5
                )
                
                # Convert recent messages to documents
                for msg in recent_messages:
                    doc = Document(
                        page_content=msg['content'],
                        metadata={
                            'source': 'recent_message',
                            'role': msg['role'],
                            'message_id': msg['message_id'],
                            'timestamp': msg['created_at'],
                            'session_id': str(self.session_id)
                        }
                    )
                    documents.append(doc)
            
            # Get relevant vectors from Pinecone
            vector_results = await vector_service.hybrid_search(
                query_embedding=query_embedding,
                user_id=self.user_id,
                session_id=self.session_id,
                message_top_k=3,
                summary_top_k=2
            )
            
            # Convert vector results to documents
            for msg_result in vector_results.get('messages', []):
                metadata = msg_result.get('metadata', {})
                doc = Document(
                    page_content=f"Historical message: {metadata.get('role', 'unknown')} message",
                    metadata={
                        'source': 'vector_message',
                        'score': msg_result.get('score', 0),
                        'vector_id': msg_result.get('id', ''),
                        **metadata
                    }
                )
                documents.append(doc)
            
            # Convert summary results to documents
            for summary_result in vector_results.get('summaries', []):
                metadata = summary_result.get('metadata', {})
                doc = Document(
                    page_content=f"Conversation summary: {metadata.get('message_count', 0)} messages summarized",
                    metadata={
                        'source': 'vector_summary',
                        'score': summary_result.get('score', 0),
                        'vector_id': summary_result.get('id', ''),
                        **metadata
                    }
                )
                documents.append(doc)
            
            # Get session summaries from Supabase
            if self.session_id:
                summaries = await supabase_service.get_session_summaries(
                    session_id=self.session_id,
                    limit=2
                )
                
                for summary in summaries:
                    doc = Document(
                        page_content=summary['summary_text'],
                        metadata={
                            'source': 'session_summary',
                            'summary_id': summary['summary_id'],
                            'message_count': summary['message_count'],
                            'timestamp': summary['created_at']
                        }
                    )
                    documents.append(doc)
            
            # Sort by relevance (recent messages first, then by vector similarity)
            documents.sort(key=lambda x: (
                x.metadata.get('source') == 'recent_message',
                x.metadata.get('score', 0)
            ), reverse=True)
            
            return documents[:self.top_k]
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise ChainError(f"Document retrieval failed: {e}")