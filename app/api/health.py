from fastapi import APIRouter, Depends
from datetime import datetime
from ..models.schemas import HealthResponse
from ..config import settings
from ..services.supabase_service import supabase_service
from ..services.vector_service import vector_service
from ..services.embedding_service import embedding_service
from ..services.memory_service import memory_service

router = APIRouter()

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Check the health of all services"""
    
    # Check individual services
    services_status = {
        "supabase": "unknown",
        "pinecone": "unknown",
        "embedding": "unknown",
        "memory": "unknown"
    }
    
    try:
        services_status["supabase"] = "healthy" if await supabase_service.health_check() else "unhealthy"
    except Exception:
        services_status["supabase"] = "unhealthy"
    
    try:
        services_status["pinecone"] = "healthy" if await vector_service.health_check() else "unhealthy"
    except Exception:
        services_status["pinecone"] = "unhealthy"
    
    try:
        services_status["embedding"] = "healthy" if await embedding_service.health_check() else "unhealthy"
    except Exception:
        services_status["embedding"] = "unhealthy"
    
    try:
        services_status["memory"] = "healthy" if await memory_service.health_check() else "unhealthy"
    except Exception:
        services_status["memory"] = "unhealthy"
    
    # Overall status
    overall_status = "healthy" if all(
        status == "healthy" for status in services_status.values()
    ) else "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        services=services_status,
        version=settings.app_version
    )

@router.get("/embedding")
async def embedding_health():
    """Check embedding service specifically"""
    return {
        "status": "healthy" if await embedding_service.health_check() else "unhealthy",
        "model_info": embedding_service.get_model_info()
    }

@router.get("/vector-stats")
async def vector_stats():
    """Get Pinecone vector database statistics"""
    try:
        # Simple health check for now
        health = await vector_service.health_check()
        return {"status": "healthy" if health else "unhealthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}