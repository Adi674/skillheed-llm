# app/services/embedding_service.py
import asyncio
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import tiktoken
import logging
from ..config import settings
from ..utils.exceptions import EmbeddingServiceError

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings using Sentence Transformers"""
    
    def __init__(self):
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self.model: Optional[SentenceTransformer] = None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # For token counting
        
    async def initialize(self):
        """Initialize the embedding model asynchronously"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                SentenceTransformer, 
                self.model_name
            )
            logger.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingServiceError(f"Failed to initialize embedding service: {e}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed, using character approximation: {e}")
            # Fallback: rough approximation
            return len(text) // 4
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        if not self.model:
            raise EmbeddingServiceError("Embedding model not initialized")
        
        if not text.strip():
            raise EmbeddingServiceError("Cannot embed empty text")
        
        try:
            # Run embedding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.model.encode,
                text
            )
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise EmbeddingServiceError(f"Embedding generation failed: {e}")
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        if not self.model:
            raise EmbeddingServiceError("Embedding model not initialized")
        
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            raise EmbeddingServiceError("No valid texts to embed")
        
        try:
            # Run batch embedding in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self.model.encode,
                valid_texts
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise EmbeddingServiceError(f"Batch embedding generation failed: {e}")
    
    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query (alias for embed_text)"""
        return await self.embed_text(query)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown') if self.model else 'not_loaded',
            "is_loaded": self.model is not None
        }
    
    async def health_check(self) -> bool:
        """Check if the embedding service is healthy"""
        try:
            if not self.model:
                return False
            
            # Test with a simple embedding
            test_embedding = await self.embed_text("health check")
            return len(test_embedding) == self.dimension
        except Exception:
            return False

# Global embedding service instance
embedding_service = EmbeddingService()