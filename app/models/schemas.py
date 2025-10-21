# app/models/schemas.py - FIXED RetrievalContext model
from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field
from .enums import MessageRole, SessionStatus

# Request Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    session_id: Optional[UUID] = Field(None, description="Session ID (creates new if not provided)")
    user_id: UUID = Field(..., description="User ID")
    session_name: Optional[str] = Field(None, max_length=255, description="Custom session name")

class SessionCreateRequest(BaseModel):
    user_id: UUID = Field(..., description="User ID")
    session_name: Optional[str] = Field("New Chat", max_length=255, description="Session name")

class SessionUpdateRequest(BaseModel):
    session_name: Optional[str] = Field(None, max_length=255)
    session_status: Optional[SessionStatus] = None

# Response Models
class MessageResponse(BaseModel):
    message_id: UUID
    session_id: UUID
    user_id: UUID
    role: MessageRole
    content: str
    token_count: Optional[int]
    created_at: datetime

class SessionResponse(BaseModel):
    session_id: UUID
    user_id: UUID
    session_name: str
    session_status: SessionStatus
    context_limit: int
    last_activity_at: Optional[datetime]
    created_at: datetime
    message_count: Optional[int] = 0

class ChatResponse(BaseModel):
    message: str
    session_id: UUID
    message_id: UUID
    token_count: Optional[int]
    response_time_ms: Optional[float]
    context_used: Optional[int]

class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total_count: int
    page: int
    page_size: int

class MemorySummaryResponse(BaseModel):
    summary_id: UUID
    session_id: UUID
    summary_text: str
    message_count: int
    created_at: datetime

# Internal Models
class VectorMetadata(BaseModel):
    type: str
    user_id: str
    session_id: str
    timestamp: str
    year: int
    month: int
    day: int

class MessageMetadata(VectorMetadata):
    message_id: str
    role: str
    token_count: int
    session_name: str

class SummaryMetadata(VectorMetadata):
    summary_id: str
    message_count: int
    start_message_id: str
    end_message_id: str
    session_name: str

# FIXED: RetrievalContext with correct data structure
class RetrievalContext(BaseModel):
    recent_messages: List[Dict[str, Any]]
    relevant_history: Dict[str, List[Dict[str, Any]]]  # FIXED: Should be a dict with 'messages' and 'summaries' keys
    summaries: List[Dict[str, Any]]
    total_tokens: int

# Error Models
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    version: str