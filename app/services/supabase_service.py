# app/services/supabase_service.py - ENHANCED VERSION
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from supabase import create_client, Client
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

from ..config import settings
from ..models.schemas import MessageResponse, SessionResponse, MemorySummaryResponse
from ..models.enums import MessageRole, SessionStatus
from ..utils.exceptions import DatabaseError

logger = logging.getLogger(__name__)

class SupabaseService:
    """Service for all Supabase database operations - ENHANCED VERSION"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self._initialized = False
        self._max_retries = 3
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def initialize(self):
        """Initialize Supabase client with retry logic and schema verification"""
        try:
            self.client = create_client(
                settings.supabase_url,
                settings.supabase_service_key
            )
            
            # Verify database schema
            await self._verify_schema()
            
            self._initialized = True
            logger.info("Supabase service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    async def _verify_schema(self):
        """Verify required tables exist and are accessible"""
        required_tables = ['users', 'sessions', 'messages', 'memory_summaries']
        
        for table in required_tables:
            try:
                # Test table access with minimal query
                result = self.client.table(table).select('*').limit(1).execute()
                logger.info(f"Table '{table}' verified successfully")
            except Exception as e:
                logger.error(f"Table '{table}' verification failed: {e}")
                raise DatabaseError(f"Required table '{table}' not found or accessible: {e}")
    
    def _ensure_initialized(self):
        """Ensure service is initialized before operations"""
        if not self._initialized or not self.client:
            raise DatabaseError("Supabase service not initialized. Call initialize() first.")
    
    # ERROR HANDLING DECORATOR
    def handle_database_errors(func):
        """Decorator to handle database errors consistently"""
        async def wrapper(self, *args, **kwargs):
            self._ensure_initialized()
            try:
                return await func(self, *args, **kwargs)
            except DatabaseError:
                raise
            except Exception as e:
                logger.error(f"Database operation failed in {func.__name__}: {e}")
                raise DatabaseError(f"Database operation failed: {e}")
        return wrapper
    
    # Session Operations
    @handle_database_errors
    async def create_session(
        self, 
        user_id: UUID, 
        session_name: str = "New Chat"
    ) -> SessionResponse:
        """Create a new chat session with proper error handling"""
        try:
            # Validate user exists first
            if not await self.verify_user_exists(user_id):
                raise DatabaseError(f"User {user_id} does not exist")
            
            session_data = {
                'user_id': str(user_id),
                'session_name': session_name,
                'session_status': SessionStatus.ACTIVE.value,
                'context_limit': settings.context_window_size
            }
            
            result = self.client.table('sessions').insert(session_data).execute()
            
            if not result.data:
                raise DatabaseError("Failed to create session - no data returned")
            
            session_data = result.data[0]
            logger.info(f"Created session {session_data.get('session_id')} for user {user_id}")
            
            return SessionResponse(**session_data)
            
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Session creation failed: {e}")
            raise DatabaseError(f"Session creation failed: {e}")
    
    @handle_database_errors
    async def get_session(self, session_id: UUID, user_id: UUID) -> Optional[SessionResponse]:
        """Get session with proper error handling"""
        try:
            result = self.client.table('sessions').select('*').eq(
                'session_id', str(session_id)
            ).eq('user_id', str(user_id)).eq(
                'session_status', SessionStatus.ACTIVE.value
            ).execute()
            
            if result.data:
                return SessionResponse(**result.data[0])
            
            logger.warning(f"Session {session_id} not found for user {user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    @handle_database_errors
    async def get_user_sessions(
        self, 
        user_id: UUID, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[SessionResponse]:
        """Get all sessions for a user"""
        try:
            result = self.client.table('sessions').select(
                '*'
            ).eq('user_id', str(user_id)).eq(
                'session_status', SessionStatus.ACTIVE.value
            ).order('last_activity_at', desc=True).range(
                offset, offset + limit - 1
            ).execute()
            
            return [SessionResponse(**session) for session in result.data]
            
        except Exception as e:
            logger.error(f"Failed to get user sessions: {e}")
            raise DatabaseError(f"Failed to retrieve sessions: {e}")
    
    async def update_session_activity(self, session_id: UUID):
        """Update session last activity timestamp"""
        try:
            self.client.table('sessions').update({
                'last_activity_at': datetime.utcnow().isoformat()
            }).eq('session_id', str(session_id)).execute()
            
        except Exception as e:
            logger.warning(f"Failed to update session activity: {e}")
    
    # Message Operations
    @handle_database_errors
    async def create_message(
        self,
        session_id: UUID,
        user_id: UUID,
        role: MessageRole,
        content: str,
        token_count: Optional[int] = None,
        embedding: Optional[List[float]] = None
    ) -> MessageResponse:
        """Create message with validation and error handling"""
        try:
            # Validate session exists
            session = await self.get_session(session_id, user_id)
            if not session:
                raise DatabaseError(f"Session {session_id} not found or not accessible")
            
            # Validate content
            if not content or len(content.strip()) == 0:
                raise DatabaseError("Message content cannot be empty")
            
            if len(content) > 4000:
                raise DatabaseError("Message content too long")
            
            message_data = {
                'session_id': str(session_id),
                'user_id': str(user_id),
                'role': role.value,
                'content': content.strip(),
                'token_count': token_count
            }
            
            if embedding:
                message_data['embedding'] = embedding
            
            result = self.client.table('messages').insert(message_data).execute()
            
            if not result.data:
                raise DatabaseError("Failed to create message - no data returned")
            
            # Update session activity
            await self.update_session_activity(session_id)
            
            message_data = result.data[0]
            logger.info(f"Created message {message_data.get('message_id')} in session {session_id}")
            
            return MessageResponse(**message_data)
            
        except DatabaseError:
            raise
        except Exception as e:
            logger.error(f"Message creation failed: {e}")
            raise DatabaseError(f"Message creation failed: {e}")
    
    async def get_session_messages(
        self,
        session_id: UUID,
        limit: Optional[int] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """Get messages for a session"""
        self._ensure_initialized()
        
        try:
            query = self.client.table('messages')
            
            if include_embeddings:
                query = query.select('*')
            else:
                query = query.select('message_id,session_id,user_id,role,content,token_count,created_at,updated_at')
            
            query = query.eq('session_id', str(session_id)).order('created_at', desc=False)
            
            if limit:
                query = query.limit(limit)
            
            result = query.execute()
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to get session messages: {e}")
            raise DatabaseError(f"Failed to retrieve messages: {e}")
    
    async def get_recent_messages(
        self,
        session_id: UUID,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent messages for context"""
        self._ensure_initialized()
        
        try:
            result = self.client.table('messages').select(
                'message_id,role,content,token_count,created_at'
            ).eq('session_id', str(session_id)).order(
                'created_at', desc=True
            ).limit(limit).execute()
            
            # Return in chronological order (oldest first)
            return list(reversed(result.data))
            
        except Exception as e:
            logger.error(f"Failed to get recent messages: {e}")
            raise DatabaseError(f"Failed to retrieve recent messages: {e}")
    
    async def update_message_pinecone_id(
        self,
        message_id: UUID,
        pinecone_id: str
    ):
        """Update message with Pinecone vector ID"""
        try:
            self.client.table('messages').update({
                'pinecone_id': pinecone_id,
                'is_stored_in_pinecone': True
            }).eq('message_id', str(message_id)).execute()
            
        except Exception as e:
            logger.error(f"Failed to update message Pinecone ID: {e}")
    
    # Memory Summary Operations
    @handle_database_errors
    async def create_memory_summary(
        self,
        session_id: UUID,
        user_id: UUID,
        summary_text: str,
        message_count: int,
        start_message_id: UUID,
        end_message_id: UUID,
        embedding: Optional[List[float]] = None
    ) -> MemorySummaryResponse:
        """Create a memory summary"""
        try:
            summary_data = {
                'session_id': str(session_id),
                'user_id': str(user_id),
                'summary_text': summary_text,
                'message_count': message_count,
                'start_message_id': str(start_message_id),
                'end_message_id': str(end_message_id)
            }
            
            if embedding:
                summary_data['embedding'] = embedding
            
            result = self.client.table('memory_summaries').insert(summary_data).execute()
            
            if not result.data:
                raise DatabaseError("Failed to create memory summary")
            
            return MemorySummaryResponse(**result.data[0])
            
        except Exception as e:
            logger.error(f"Failed to create memory summary: {e}")
            raise DatabaseError(f"Memory summary creation failed: {e}")
    
    async def get_session_summaries(
        self,
        session_id: UUID,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get memory summaries for a session"""
        self._ensure_initialized()
        
        try:
            result = self.client.table('memory_summaries').select(
                '*'
            ).eq('session_id', str(session_id)).order(
                'created_at', desc=True
            ).limit(limit).execute()
            
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to get session summaries: {e}")
            raise DatabaseError(f"Failed to retrieve summaries: {e}")
    
    async def count_session_messages(self, session_id: UUID) -> int:
        """Count total messages in a session"""
        self._ensure_initialized()
        
        try:
            result = self.client.table('messages').select(
                'message_id', count='exact'
            ).eq('session_id', str(session_id)).execute()
            
            return result.count or 0
            
        except Exception as e:
            logger.error(f"Failed to count session messages: {e}")
            return 0
    
    # User Operations
    async def get_user(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get user"""
        self._ensure_initialized()
        
        try:
            result = self.client.table('users').select('*').eq(
                'user_id', str(user_id)
            ).single().execute()
            
            return result.data if result.data else None
            
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    async def verify_user_exists(self, user_id: UUID) -> bool:
        """Verify user exists"""
        self._ensure_initialized()
        
        try:
            result = self.client.table('users').select('user_id').eq(
                'user_id', str(user_id)
            ).single().execute()
            
            return result.data is not None
            
        except Exception as e:
            logger.error(f"Failed to verify user: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Enhanced health check with actual database operation"""
        try:
            if not self._initialized or not self.client:
                return False
            
            # Test with actual query
            result = self.client.table('users').select('user_id').limit(1).execute()
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Global Supabase service instance
supabase_service = SupabaseService()