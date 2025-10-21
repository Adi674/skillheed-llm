class ChatbotError(Exception):
    """Base exception for chatbot errors"""
    pass

class DatabaseError(ChatbotError):
    """Database operation errors"""
    pass

class VectorServiceError(ChatbotError):
    """Vector database operation errors"""
    pass

class EmbeddingServiceError(ChatbotError):
    """Embedding generation errors"""
    pass

class MemoryServiceError(ChatbotError):
    """Memory management errors"""
    pass

class ChainError(ChatbotError):
    """LangChain operation errors"""
    pass

class ValidationError(ChatbotError):
    """Input validation errors"""
    pass
