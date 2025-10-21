# app/config.py - DEBUG AND FIX VERSION
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    app_name: str = "RAG Chatbot API"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Supabase Settings - FIXED: Use correct env variable name
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_service_key: str = Field(..., env="SUPABASE_SERVICE_KEY")  # Changed to match .env
    
    # Pinecone Settings
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_environment: str = Field(..., env="PINECONE_ENVIRONMENT")
    messages_index_name: str = Field(default="chatbot-messages", env="MESSAGES_INDEX_NAME")
    summaries_index_name: str = Field(default="chatbot-summaries", env="SUMMARIES_INDEX_NAME")
    
    # Groq Settings
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", env="GROQ_MODEL")
    groq_temperature: float = Field(default=0.7, env="GROQ_TEMPERATURE")
    groq_max_tokens: int = Field(default=1024, env="GROQ_MAX_TOKENS")
    groq_streaming: bool = Field(default=False, env="GROQ_STREAMING")
    
    # Embedding Settings
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    
    # Memory Settings
    max_tokens_per_session: int = Field(default=8000, env="MAX_TOKENS_PER_SESSION")
    context_window_size: int = Field(default=15, env="CONTEXT_WINDOW_SIZE")
    summary_trigger_threshold: int = Field(default=25, env="SUMMARY_TRIGGER_THRESHOLD")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Optional settings (with defaults to avoid validation errors)
    llama_system_prompt: str = Field(
        default="You are a helpful AI assistant. Provide accurate, relevant, and concise responses based on the conversation context and retrieved information.",
        env="LLAMA_SYSTEM_PROMPT"
    )
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    vector_search_top_k: int = Field(default=5, env="VECTOR_SEARCH_TOP_K")
    memory_summary_max_length: int = Field(default=500, env="MEMORY_SUMMARY_MAX_LENGTH")
    context_truncation_limit: int = Field(default=50000, env="CONTEXT_TRUNCATION_LIMIT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields to prevent validation errors

# Global settings instance
settings = Settings()

# Model information for Llama 3.3 70B Versatile
LLAMA_MODEL_INFO = {
    "name": "llama-3.3-70b-versatile",
    "max_context_length": 131072,
    "recommended_max_tokens": 1024,
    "temperature_range": (0.0, 2.0),
    "supports_streaming": True,
    "optimal_use_cases": [
        "Complex reasoning",
        "Long-form content generation", 
        "Code generation and debugging",
        "Multi-turn conversations",
        "Creative writing"
    ]
}