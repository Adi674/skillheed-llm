# app/services/vector_service.py - FIXED PINECONE IMPORTS
from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from datetime import datetime
import logging

# CORRECT Pinecone imports for the new package
from pinecone import Pinecone, ServerlessSpec

from ..config import settings
from ..utils.exceptions import VectorServiceError

logger = logging.getLogger(__name__)

class VectorService:
    """Service for Pinecone vector database operations"""
    
    def __init__(self):
        self.pc: Optional[Pinecone] = None
        self.messages_index = None
        self.summaries_index = None
    
    async def initialize(self):
        """Initialize Pinecone client and indexes"""
        try:
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            
            # Create indexes if they don't exist
            await self._ensure_indexes_exist()
            
            # Connect to indexes
            self.messages_index = self.pc.Index(settings.messages_index_name)
            self.summaries_index = self.pc.Index(settings.summaries_index_name)
            
            logger.info("Pinecone client and indexes initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise VectorServiceError(f"Vector service initialization failed: {e}")
    
    async def _ensure_indexes_exist(self):
        """Create Pinecone indexes if they don't exist"""
        existing_indexes = self.pc.list_indexes().names()
        
        # Create messages index
        if settings.messages_index_name not in existing_indexes:
            logger.info(f"Creating messages index: {settings.messages_index_name}")
            self.pc.create_index(
                name=settings.messages_index_name,
                dimension=settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        # Create summaries index
        if settings.summaries_index_name not in existing_indexes:
            logger.info(f"Creating summaries index: {settings.summaries_index_name}")
            self.pc.create_index(
                name=settings.summaries_index_name,
                dimension=settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
    
    # Message Vector Operations
    def _create_message_vector_id(
        self, 
        user_id: UUID, 
        session_id: UUID, 
        message_id: UUID
    ) -> str:
        """Create standardized vector ID for messages"""
        return f"msg_{user_id}_{session_id}_{message_id}"
    
    def _create_message_metadata(
        self,
        user_id: UUID,
        session_id: UUID,
        message_id: UUID,
        role: str,
        timestamp: datetime,
        token_count: int,
        session_name: str = "chat"
    ) -> Dict[str, Any]:
        """Create metadata for message vectors"""
        return {
            "type": "message",
            "user_id": str(user_id),
            "session_id": str(session_id),
            "message_id": str(message_id),
            "role": role,
            "timestamp": timestamp.isoformat(),
            "token_count": token_count,
            "session_name": session_name,
            "year": timestamp.year,
            "month": timestamp.month,
            "day": timestamp.day
        }
    
    async def store_message_vector(
        self,
        user_id: UUID,
        session_id: UUID,
        message_id: UUID,
        embedding: List[float],
        role: str,
        token_count: int,
        timestamp: datetime,
        session_name: str = "chat"
    ) -> str:
        """Store message embedding in Pinecone"""
        try:
            vector_id = self._create_message_vector_id(user_id, session_id, message_id)
            metadata = self._create_message_metadata(
                user_id, session_id, message_id, role, 
                timestamp, token_count, session_name
            )
            
            self.messages_index.upsert([{
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }])
            
            logger.info(f"Stored message vector: {vector_id}")
            return vector_id
        except Exception as e:
            logger.error(f"Failed to store message vector: {e}")
            raise VectorServiceError(f"Message vector storage failed: {e}")
    
    async def search_message_vectors(
        self,
        query_embedding: List[float],
        user_id: UUID,
        session_id: Optional[UUID] = None,
        top_k: int = 5,
        time_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant message vectors"""
        try:
            # Build filter
            filter_dict = {
                "type": "message",
                "user_id": str(user_id)
            }
            
            if session_id:
                filter_dict["session_id"] = str(session_id)
            
            if time_filter:
                filter_dict.update(time_filter)
            
            # Search vectors
            results = self.messages_index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            return [{
                "id": match["id"],
                "score": match["score"],
                "metadata": match["metadata"]
            } for match in results["matches"]]
            
        except Exception as e:
            logger.error(f"Failed to search message vectors: {e}")
            raise VectorServiceError(f"Message vector search failed: {e}")
    
    # Summary Vector Operations
    def _create_summary_vector_id(
        self, 
        user_id: UUID, 
        session_id: UUID, 
        summary_id: UUID
    ) -> str:
        """Create standardized vector ID for summaries"""
        return f"sum_{user_id}_{session_id}_{summary_id}"
    
    def _create_summary_metadata(
        self,
        user_id: UUID,
        session_id: UUID,
        summary_id: UUID,
        message_count: int,
        start_message_id: UUID,
        end_message_id: UUID,
        timestamp: datetime,
        session_name: str = "chat"
    ) -> Dict[str, Any]:
        """Create metadata for summary vectors"""
        return {
            "type": "summary",
            "user_id": str(user_id),
            "session_id": str(session_id),
            "summary_id": str(summary_id),
            "message_count": message_count,
            "start_message_id": str(start_message_id),
            "end_message_id": str(end_message_id),
            "timestamp": timestamp.isoformat(),
            "session_name": session_name,
            "year": timestamp.year,
            "month": timestamp.month
        }
    
    async def store_summary_vector(
        self,
        user_id: UUID,
        session_id: UUID,
        summary_id: UUID,
        embedding: List[float],
        message_count: int,
        start_message_id: UUID,
        end_message_id: UUID,
        timestamp: datetime,
        session_name: str = "chat"
    ) -> str:
        """Store summary embedding in Pinecone"""
        try:
            vector_id = self._create_summary_vector_id(user_id, session_id, summary_id)
            metadata = self._create_summary_metadata(
                user_id, session_id, summary_id, message_count,
                start_message_id, end_message_id, timestamp, session_name
            )
            
            self.summaries_index.upsert([{
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }])
            
            logger.info(f"Stored summary vector: {vector_id}")
            return vector_id
        except Exception as e:
            logger.error(f"Failed to store summary vector: {e}")
            raise VectorServiceError(f"Summary vector storage failed: {e}")
    
    async def search_summary_vectors(
        self,
        query_embedding: List[float],
        user_id: UUID,
        session_id: Optional[UUID] = None,
        top_k: int = 3,
        exclude_current_session: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for relevant summary vectors"""
        try:
            # Build filter
            filter_dict = {
                "type": "summary",
                "user_id": str(user_id)
            }
            
            if exclude_current_session and session_id:
                # Exclude current session to get cross-session context
                filter_dict["session_id"] = {"$ne": str(session_id)}
            elif session_id:
                filter_dict["session_id"] = str(session_id)
            
            # Search vectors
            results = self.summaries_index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            return [{
                "id": match["id"],
                "score": match["score"],
                "metadata": match["metadata"]
            } for match in results["matches"]]
            
        except Exception as e:
            logger.error(f"Failed to search summary vectors: {e}")
            raise VectorServiceError(f"Summary vector search failed: {e}")
    
    # Combined Search Operations
    async def hybrid_search(
        self,
        query_embedding: List[float],
        user_id: UUID,
        session_id: Optional[UUID] = None,
        message_top_k: int = 3,
        summary_top_k: int = 2
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform hybrid search across messages and summaries"""
        try:
            # Search messages and summaries in parallel
            import asyncio
            
            message_task = self.search_message_vectors(
                query_embedding, user_id, session_id, message_top_k
            )
            summary_task = self.search_summary_vectors(
                query_embedding, user_id, session_id, summary_top_k
            )
            
            message_results, summary_results = await asyncio.gather(
                message_task, summary_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(message_results, Exception):
                logger.error(f"Message search failed: {message_results}")
                message_results = []
            
            if isinstance(summary_results, Exception):
                logger.error(f"Summary search failed: {summary_results}")
                summary_results = []
            
            return {
                "messages": message_results,
                "summaries": summary_results
            }
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise VectorServiceError(f"Hybrid search failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if vector service is healthy"""
        try:
            if not self.messages_index or not self.summaries_index:
                return False
            
            # Test basic operations
            stats = self.pc.list_indexes()
            return len(stats.names()) > 0
        except Exception:
            return False

# Global vector service instance
vector_service = VectorService()