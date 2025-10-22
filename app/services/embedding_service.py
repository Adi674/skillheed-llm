# app/services/embedding_service.py - OPTIMIZED WITH BATCH PROCESSING
import asyncio
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import tiktoken

from ..config import settings
from ..utils.exceptions import EmbeddingServiceError
from .batch_processor import AsyncBatchProcessor

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Optimized Embedding Service with batch processing capabilities"""
    
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.tokenizer = None
        self._initialized = False
        
        # Batch processor for embedding operations
        self.batch_processor = None
        
        # Performance tracking
        self._total_embeddings_generated = 0
        self._batch_count = 0
    
    async def initialize(self):
        """Initialize embedding service with optimized batch processing"""
        try:
            # Initialize sentence transformer model
            self.model = SentenceTransformer(settings.embedding_model)
            
            # Warm up the model with a test embedding
            logger.info("Warming up embedding model...")
            _ = self.model.encode(["warmup test"])
            logger.info("Model warmed up successfully")
            
            # Initialize tokenizer
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer: {e}")
                self.tokenizer = None
            
            # Initialize batch processor with optimized settings
            self.batch_processor = AsyncBatchProcessor(
                batch_size=32,  # Reduced batch size for faster processing
                max_wait_time=0.01,  # Reduced wait time to 10ms
                max_concurrent_batches=3,  # Reduced concurrent batches
                processor_func=self._process_embedding_batch
            )
            
            self._initialized = True
            logger.info(f"Embedding service initialized with model: {settings.embedding_model}")
            logger.info(f"Embedding dimension: {settings.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise EmbeddingServiceError(f"Embedding service initialization failed: {e}")
    
    def _ensure_initialized(self):
        """Ensure service is initialized before operations"""
        if not self._initialized or not self.model:
            raise EmbeddingServiceError("Embedding service not initialized. Call initialize() first.")
    
    async def _process_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        """Process batch of embeddings efficiently using thread pool"""
        try:
            # Validate inputs
            if not texts:
                return []
            
            # Filter out empty texts and normalize
            valid_texts = []
            for text in texts:
                cleaned_text = text.strip() if text else ""
                if cleaned_text:
                    valid_texts.append(cleaned_text)
                else:
                    valid_texts.append("empty")  # Placeholder for empty texts
            
            # Process in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self.model.encode,
                valid_texts
            )
            
            # Update statistics
            self._total_embeddings_generated += len(valid_texts)
            self._batch_count += 1
            
            logger.debug(f"Generated {len(valid_texts)} embeddings in batch #{self._batch_count}")
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Batch embedding processing failed: {e}")
            raise EmbeddingServiceError(f"Batch embedding generation failed: {e}")
    
    # NEW OPTIMIZED METHOD: Batch embedding with automatic queueing
    async def embed_text_batch(self, text: str, user_id: str = "unknown") -> List[float]:
        """
        Embed text using batch processor for optimal performance.
        This method automatically queues requests and processes them in batches.
        """
        self._ensure_initialized()
        
        if not text or not text.strip():
            raise EmbeddingServiceError("Empty text provided for embedding")
        
        try:
            # Submit to batch processor for optimized processing
            embedding = await self.batch_processor.submit(text.strip(), user_id)
            return embedding
            
        except Exception as e:
            logger.error(f"Batch embedding failed for user {user_id}: {e}")
            # Fallback to direct processing
            logger.info("Falling back to direct embedding processing")
            return await self.embed_text(text)
    
    # OPTIMIZED METHOD: Manual batch processing for multiple texts
    async def embed_batch_manual(self, texts: List[str]) -> List[List[float]]:
        """
        Process multiple texts in one batch call.
        Use this when you have multiple texts to process at once.
        """
        self._ensure_initialized()
        
        if not texts:
            return []
        
        try:
            # Process all texts in one batch operation
            embeddings = await self._process_embedding_batch(texts)
            logger.info(f"Manual batch processed {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Manual batch embedding failed: {e}")
            raise EmbeddingServiceError(f"Manual batch embedding failed: {e}")
    
    # OPTIMIZED METHOD: Chunked processing for very large batches
    async def embed_batch_chunked(self, texts: List[str], chunk_size: int = 128) -> List[List[float]]:
        """
        Process very large batches by breaking them into chunks.
        Useful for processing hundreds or thousands of texts efficiently.
        """
        self._ensure_initialized()
        
        if not texts:
            return []
        
        if len(texts) <= chunk_size:
            return await self.embed_batch_manual(texts)
        
        try:
            # Process in chunks for memory efficiency
            all_embeddings = []
            tasks = []
            
            # Create tasks for each chunk
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                task = asyncio.create_task(self.embed_batch_manual(chunk))
                tasks.append(task)
            
            # Wait for all chunks to complete
            chunk_results = await asyncio.gather(*tasks)
            
            # Flatten results
            for chunk_embeddings in chunk_results:
                all_embeddings.extend(chunk_embeddings)
            
            logger.info(f"Chunked processing completed: {len(texts)} texts in {len(tasks)} chunks")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Chunked batch embedding failed: {e}")
            raise EmbeddingServiceError(f"Chunked batch processing failed: {e}")
    
    # ORIGINAL METHOD: Direct embedding (kept for compatibility)
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text directly.
        Note: For better performance, use embed_text_batch() instead.
        """
        self._ensure_initialized()
        
        try:
            # Clean and validate input
            text = text.strip()
            if not text:
                raise EmbeddingServiceError("Empty text provided")
            
            # Process in thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.model.encode,
                [text]
            )
            
            self._total_embeddings_generated += 1
            return embedding[0].tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise EmbeddingServiceError(f"Embedding generation failed: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tokenizer"""
        try:
            if not text:
                return 0
            
            if self.tokenizer:
                return len(self.tokenizer.encode(text))
            else:
                # Fallback: rough estimation (4 characters per token)
                return max(1, len(text.strip()) // 4)
                
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback estimation
            return max(1, len(text.strip()) // 4)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this service"""
        return settings.embedding_dimension
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model being used"""
        return settings.embedding_model
    
    def get_statistics(self) -> dict:
        """Get performance statistics"""
        stats = {
            "total_embeddings_generated": self._total_embeddings_generated,
            "total_batches_processed": self._batch_count,
            "embedding_dimension": settings.embedding_dimension,
            "model_name": settings.embedding_model,
            "initialized": self._initialized
        }
        
        # Add batch processor stats if available
        if self.batch_processor:
            try:
                batch_stats = self.batch_processor.get_stats()
                stats.update({
                    "batch_processor": batch_stats
                })
            except Exception:
                pass
        
        return stats
    
    async def health_check(self) -> bool:
        """Check if embedding service is healthy"""
        try:
            if not self._initialized or not self.model:
                return False
            
            # Test embedding generation with a simple text
            test_embedding = await self.embed_text("health check test")
            
            # Verify embedding has correct dimension
            if len(test_embedding) != settings.embedding_dimension:
                logger.error(f"Health check failed: wrong dimension {len(test_embedding)}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False
    
    async def warm_up(self):
        """Warm up the service by generating a test embedding"""
        try:
            logger.info("Warming up embedding service...")
            await self.embed_text("warmup test")
            logger.info("Embedding service warmed up successfully")
        except Exception as e:
            logger.warning(f"Embedding service warmup failed: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.batch_processor:
                # Note: AsyncBatchProcessor doesn't have cleanup method in our implementation
                # but we can add this for future use
                pass
            
            self._initialized = False
            logger.info("Embedding service cleaned up")
            
        except Exception as e:
            logger.error(f"Embedding service cleanup failed: {e}")

# Global embedding service instance
embedding_service = EmbeddingService()