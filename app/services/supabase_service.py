# app/services/supabase_service.py
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from supabase import create_client, Client
import logging
from ..config import settings
from ..models.schemas import MessageResponse, SessionResponse, MemorySummaryResponse
from ..models.enums import MessageRole, SessionStatus
from ..utils.exceptions import DatabaseError

logger = logging.getLogger(__name__)

class SupabaseService:
    """Service for all Supabase database operations - connects to existing database"""
    
    def __init__(self):
        self.client: Optional[Client] = None
    
    async def initialize(self):
        """Initialize Supabase client to connect to existing database"""
        try:
            self.client = create_client(
                settings.supabase_url,
                settings.supabase_service_key  # Updated to match config field name
            )
            
            # Test connection with existing users table
            test_query = self.client.table('users').select('user_id').limit(1).execute()
            logger.info("Supabase client connected to existing database successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Supabase database: {e}")
            raise DatabaseError(f"Database connection failed: {e}")
    
    # Session Operations - Work with your existing sessions table
    async def create_session(
        self, 
        user_id: UUID, 
        session_name: str = "New Chat"
    ) -> SessionResponse:
        """Create a new chat session in existing sessions table"""
        try:
            result = self.client.table('sessions').insert({
                'user_id': str(user_id),
                'session_name': session_name,
                'session_status': SessionStatus.ACTIVE.value,
                'context_limit': settings.context_window_size
            }).execute()
            
            if not result.data:
                raise DatabaseError("Failed to create session")
            
            session_data = result.data[0]
            return SessionResponse(**session_data)
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise DatabaseError(f"Session creation failed: {e}")
    
    async def get_session(self, session_id: UUID, user_id: UUID) -> Optional[SessionResponse]:
        """Get session by ID and user ID from existing sessions table"""
        try:
            result = self.client.table('sessions').select('*').eq(
                'session_id', str(session_id)
            ).eq('user_id', str(user_id)).eq(
                'session_status', SessionStatus.ACTIVE.value
            ).single().execute()
            
            if result.data:
                return SessionResponse(**result.data)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    async def get_user_sessions(
        self, 
        user_id: UUID, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[SessionResponse]:
        """Get all sessions for a user from existing sessions table"""
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
    
    # Message Operations - Work with your existing messages table
    async def create_message(
        self,
        session_id: UUID,
        user_id: UUID,
        role: MessageRole,
        content: str,
        token_count: Optional[int] = None,
        embedding: Optional[List[float]] = None
    ) -> MessageResponse:
        """Create a new message in existing messages table"""
        try:
            message_data = {
                'session_id': str(session_id),
                'user_id': str(user_id),
                'role': role.value,
                'content': content,
                'token_count': token_count
            }
            
            if embedding:
                message_data['embedding'] = embedding
            
            result = self.client.table('messages').insert(message_data).execute()
            
            if not result.data:
                raise DatabaseError("Failed to create message")
            
            # Update session activity (handled by database trigger)
            await self.update_session_activity(session_id)
            
            return MessageResponse(**result.data[0])
            
        except Exception as e:
            logger.error(f"Failed to create message: {e}")
            raise DatabaseError(f"Message creation failed: {e}")
    
    async def get_session_messages(
        self,
        session_id: UUID,
        limit: Optional[int] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """Get messages for a session from existing messages table"""
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
        """Get recent messages for context from existing messages table"""
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
        """Update message with Pinecone vector ID in existing messages table"""
        try:
            self.client.table('messages').update({
                'pinecone_id': pinecone_id,
                'is_stored_in_pinecone': True
            }).eq('message_id', str(message_id)).execute()
            
        except Exception as e:
            logger.error(f"Failed to update message Pinecone ID: {e}")
    
    # Memory Summary Operations - Work with your existing memory_summaries table
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
        """Create a memory summary in existing memory_summaries table"""
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
        """Get memory summaries for a session from existing memory_summaries table"""
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
        """Count total messages in a session from existing messages table"""
        try:
            result = self.client.table('messages').select(
                'message_id', count='exact'
            ).eq('session_id', str(session_id)).execute()
            
            return result.count or 0
            
        except Exception as e:
            logger.error(f"Failed to count session messages: {e}")
            return 0
    
    # User Operations - Work with your existing users table
    async def get_user(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get user from existing users table"""
        try:
            result = self.client.table('users').select('*').eq(
                'user_id', str(user_id)
            ).single().execute()
            
            return result.data if result.data else None
            
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    async def verify_user_exists(self, user_id: UUID) -> bool:
        """Verify user exists in existing users table"""
        try:
            result = self.client.table('users').select('user_id').eq(
                'user_id', str(user_id)
            ).single().execute()
            
            return result.data is not None
            
        except Exception as e:
            logger.error(f"Failed to verify user: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if Supabase connection is healthy by querying existing users table"""
        try:
            # Simple query to test connection with existing table
            result = self.client.table('users').select('user_id').limit(1).execute()
            return True
            
        except Exception:
            return False

# Global Supabase service instance
supabase_service = SupabaseService()