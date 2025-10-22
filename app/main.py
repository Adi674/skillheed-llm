# app/main.py - FIXED IMPORTS
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from .config import settings
from .services.supabase_service import supabase_service
from .services.vector_service import vector_service
from .services.embedding_service import embedding_service
from .services.memory_service import memory_service

from .api.chat import router as chat_router
from .api.sessions import router as sessions_router
from .api.health import router as health_router

from .utils.exceptions import ChatbotError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    logger.info("Starting RAG Chatbot API...")
    
    try:
        # Initialize services
        await embedding_service.initialize()
        await supabase_service.initialize()
        await vector_service.initialize()
        
        logger.info("All services initialized successfully")
        
        # Store services in app state for access in routes
        app.state.embedding_service = embedding_service
        app.state.supabase_service = supabase_service
        app.state.vector_service = vector_service
        app.state.memory_service = memory_service
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Chatbot API...")

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG-powered chatbot API with LangChain, Groq, Supabase, and Pinecone",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(ChatbotError)
async def chatbot_error_handler(request: Request, exc: ChatbotError):
    logger.error(f"Chatbot error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

# Include routers
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat"])
app.include_router(sessions_router, prefix="/api/v1/sessions", tags=["Sessions"])
# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot API",
        "version": settings.app_version,
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )