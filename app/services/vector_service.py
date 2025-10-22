# app/services/vector_service.py - OPTIMIZED WITH BATCH PROCESSING
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
import logging
import asyncio

try:
    from pinecone import Pinecone, ServerlessSpec, PodSpec
except ImportError:
    import pinecone
    Pinecone = pinecone.Pinecone
    ServerlessSpec = pinecone.ServerlessSpec

from ..config import settings
from ..utils.exceptions import VectorServiceError
from .batch_processor import AsyncBatchProcessor

logger = logging.getLogger(__name__)

class VectorService:
    """Optimized Vector Service with batch processing and connection pooling"""
    
    def __init__(self):
        self.pc: Optional[Pinecone] = None
        self.messages_index = None
        self.summaries_index = None
        self._initialized = False
        self._max_retries = 3
        
        # Batch processor for vector operations
        self.vector_processor = None
        
        # Semaphore to limit concurrent operations (prevents overwhelming Pinecone)
        self._semaphore = asyncio.Semaphore(20)  # Max 20 concurrent vector operations
    
    async def initialize(self):
        """Initialize with batch processing capabilities"""
        for attempt in range(self._max_retries):
            try:
                # Initialize Pinecone client
                self.pc = Pinecone(api_key=settings.pinecone_api_key)
                logger.info("Pinecone client initialized")
                
                # Get existing indexes
                existing_indexes = self._get_existing_indexes()
                logger.info(f"Existing indexes: {existing_indexes}")
                
                # Create indexes if they don't exist
                await self._ensure_indexes_exist(existing_indexes)
                
                # Wait for indexes to be ready
                await asyncio.sleep(2)
                
                # Connect to indexes
                self.messages_index = self.pc.Index(settings.messages_index_name)
                self.summaries_index = self.pc.Index(settings.summaries_index_name)
                
                # Verify connections
                await self._verify_indexes()
                
                # Initialize batch processor for vector operations
                self.vector_processor = AsyncBatchProcessor(
                    batch_size=32,  # Process 32 vectors at once
                    max_wait_time=0.05,  # Wait max 50ms
                    max_concurrent_batches=8,  # Max 8 concurrent batches
                    processor_func=self._process_vector_batch
                )
                
                self._initialized = True
                logger.info("Vector service with batch processing initialized successfully")
                return
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed to initialize Pinecone: {e}")
                if attempt == self._max_retries - 1:
                    raise VectorServiceError(f"Vector service initialization failed: {e}")
                await asyncio.sleep(2 ** attempt)
    
    def _get_existing_indexes(self) -> List[str]:
        """Get list of existing index names"""
        try:
            indexes_response = self.pc.list_indexes()
            if hasattr(indexes_response, 'names'):
                return indexes_response.names()
            elif hasattr(indexes_response, 'get'):
                return [idx.get('name') for idx in indexes_response.get('indexes', [])]
            else:
                return [idx['name'] for idx in indexes_response['indexes']]
        except Exception as e:
            logger.warning(f"Could not list indexes: {e}")
            return []
    
    async def _ensure_indexes_exist(self, existing_indexes: List[str]):
        """Create indexes if they don't exist"""
        spec = self._get_index_spec()
        
        indexes_to_create = [
            (settings.messages_index_name, "messages"),
            (settings.summaries_index_name, "summaries")
        ]
        
        for index_name, index_type in indexes_to_create:
            if index_name not in existing_indexes:
                logger.info(f"Creating {index_type} index: {index_name}")
                try:
                    self.pc.create_index(
                        name=index_name,
                        dimension=settings.embedding_dimension,
                        metric="cosine",
                        spec=spec
                    )
                    logger.info(f"Successfully created index: {index_name}")
                    await asyncio.sleep(5)
                except Exception as e:
                    logger.error(f"Failed to create index {index_name}: {e}")
                    raise VectorServiceError(f"Index creation failed: {e}")
            else:
                logger.info(f"Index {index_name} already exists")
    
    def _get_index_spec(self):
        """Get appropriate index spec"""
        try:
            return ServerlessSpec(cloud='aws', region='us-east-1')
        except Exception:
            logger.warning("Using default serverless spec")
            return ServerlessSpec(cloud='aws', region='us-east-1')
    
    async def _verify_indexes(self):
        """Verify index connections work"""
        try:
            stats1 = self.messages_index.describe_index_stats()
            stats2 = self.summaries_index.describe_index_stats()
            
            logger.info(f"Messages index: {stats1.get('total_vector_count', 0)} vectors")
            logger.info(f"Summaries index: {stats2.get('total_vector_count', 0)} vectors")
            
        except Exception as e:
            logger.error(f"Index verification failed: {e}")
            raise VectorServiceError(f"Index verification failed: {e}")
    
    def _check_initialized(self):
        """Check if service is initialized"""
        if not self._initialized:
            raise VectorServiceError("Vector service not initialized")
    
    # BATCH PROCESSING METHODS
    async def _process_vector_batch(self, vector_data_list: List[Dict]) -> List[str]:
        """Process multiple vector operations in parallel with connection limiting"""
        try:
            # Create semaphore-limited tasks for parallel execution
            async def limited_store(vector_data):
                async with self._semaphore:  # Limit concurrent connections
                    return await self._store_single_vector(vector_data)
            
            # Process all vectors in parallel with concurrency control
            tasks = [limited_store(data) for data in vector_data_list]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle results and exceptions
            vector_ids = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Vector storage {i} failed: {result}")
                    vector_ids.append(f"error_{i}")
                else:
                    vector_ids.append(result)
            
            logger.info(f"Processed {len(vector_data_list)} vectors in batch")
            return vector_ids
            
        except Exception as e:
            logger.error(f"Batch vector processing failed: {e}")
            raise VectorServiceError(f"Batch processing failed: {e}")
    
    async def _store_single_vector(self, vector_data: Dict) -> str:
        """Store single vector (used by batch processor)"""
        return await self.store_message_vector(**vector_data)
    
    # NEW BATCH METHODS
    async def store_message_vector_batch(
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
        """Store vector using batch processor (FASTER)"""
        vector_data = {
            'user_id': user_id,
            'session_id': session_id,
            'message_id': message_id,
            'embedding': embedding,
            'role': role,
            'token_count': token_count,
            'timestamp': timestamp,
            'session_name': session_name
        }
        
        try:
            # Submit to batch processor for optimized processing
            return await self.vector_processor.submit(vector_data, str(user_id))
        except Exception as e:
            logger.error(f"Batch vector storage failed: {e}")
            # Fallback to direct storage
            return await self.store_message_vector(**vector_data)
    
    # EXISTING METHODS (unchanged)
    def _create_message_vector_id(self, user_id: UUID, session_id: UUID, message_id: UUID) -> str:
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
        """Store message embedding in Pinecone (original method)"""
        self._check_initialized()
        
        try:
            vector_id = self._create_message_vector_id(user_id, session_id, message_id)
            metadata = self._create_message_metadata(
                user_id, session_id, message_id, role, 
                timestamp, token_count, session_name
            )
            
            if len(embedding) != settings.embedding_dimension:
                raise VectorServiceError(f"Embedding dimension mismatch")
            
            self.messages_index.upsert(vectors=[{
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
        self._check_initialized()
        
        try:
            filter_dict = {
                "type": {"$eq": "message"},
                "user_id": {"$eq": str(user_id)}
            }
            
            if session_id:
                filter_dict["session_id"] = {"$eq": str(session_id)}
            
            if time_filter:
                filter_dict.update(time_filter)
            
            if len(query_embedding) != settings.embedding_dimension:
                raise VectorServiceError(f"Query embedding dimension mismatch")
            
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
            } for match in results.get("matches", [])]
            
        except Exception as e:
            logger.error(f"Failed to search message vectors: {e}")
            raise VectorServiceError(f"Message vector search failed: {e}")
    
    # Summary vector methods (unchanged)
    def _create_summary_vector_id(self, user_id: UUID, session_id: UUID, summary_id: UUID) -> str:
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
        self._check_initialized()
        
        try:
            vector_id = self._create_summary_vector_id(user_id, session_id, summary_id)
            metadata = self._create_summary_metadata(
                user_id, session_id, summary_id, message_count,
                start_message_id, end_message_id, timestamp, session_name
            )
            
            if len(embedding) != settings.embedding_dimension:
                raise VectorServiceError(f"Embedding dimension mismatch")
            
            self.summaries_index.upsert(vectors=[{
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
        self._check_initialized()
        
        try:
            filter_dict = {
                "type": {"$eq": "summary"},
                "user_id": {"$eq": str(user_id)}
            }
            
            if exclude_current_session and session_id:
                filter_dict["session_id"] = {"$ne": str(session_id)}
            elif session_id:
                filter_dict["session_id"] = {"$eq": str(session_id)}
            
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
            } for match in results.get("matches", [])]
            
        except Exception as e:
            logger.error(f"Failed to search summary vectors: {e}")
            raise VectorServiceError(f"Summary vector search failed: {e}")
    
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
            message_task = self.search_message_vectors(
                query_embedding, user_id, session_id, message_top_k
            )
            summary_task = self.search_summary_vectors(
                query_embedding, user_id, session_id, summary_top_k
            )
            
            message_results, summary_results = await asyncio.gather(
                message_task, summary_task, return_exceptions=True
            )
            
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
            return {"messages": [], "summaries": []}
    
    async def health_check(self) -> bool:
        """Check if vector service is healthy"""
        try:
            if not self._initialized or not self.pc:
                return False
            
            indexes = self._get_existing_indexes()
            return (
                settings.messages_index_name in indexes and 
                settings.summaries_index_name in indexes
            )
        except Exception:
            return False

# Global vector service instance
vector_service = VectorService()