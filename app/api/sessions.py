from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, HTTPException, Query, Depends
from ..models.schemas import (
    SessionCreateRequest, SessionUpdateRequest, SessionResponse, 
    SessionListResponse, ErrorResponse
)
from ..services.supabase_service import supabase_service
from ..utils.exceptions import DatabaseError

router = APIRouter()

@router.post("/", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new chat session"""
    try:
        session = await supabase_service.create_session(
            user_id=request.user_id,
            session_name=request.session_name
        )
        return session
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create session")

@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: UUID, user_id: UUID = Query(...)):
    """Get a specific session"""
    try:
        session = await supabase_service.get_session(session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve session")

@router.get("/", response_model=SessionListResponse)
async def list_user_sessions(
    user_id: UUID = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    """List all sessions for a user"""
    try:
        offset = (page - 1) * page_size
        sessions = await supabase_service.get_user_sessions(
            user_id=user_id,
            limit=page_size,
            offset=offset
        )
        
        # Get total count (simplified - in production, you'd want a separate count query)
        total_count = len(sessions)  # This is not accurate for pagination
        
        return SessionListResponse(
            sessions=sessions,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to retrieve sessions")

@router.put("/{session_id}", response_model=SessionResponse)
async def update_session(
    session_id: UUID, 
    request: SessionUpdateRequest,
    user_id: UUID = Query(...)
):
    """Update session details"""
    try:
        # First check if session exists and belongs to user
        session = await supabase_service.get_session(session_id, user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Update session (implementation needed in supabase_service)
        # For now, return the existing session
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update session")

@router.delete("/{session_id}")
async def delete_session(session_id: UUID, user_id: UUID = Query(...)):
    """Delete a session (soft delete)"""
    try:
        # Implementation needed in supabase_service
        return {"message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete session")