from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain.schema import BaseRetriever, Document
from uuid import UUID

class BaseCustomRetriever(BaseRetriever, ABC):
    """Base class for custom retrievers"""
    
    def __init__(self, user_id: UUID, session_id: UUID = None):
        super().__init__()
        self.user_id = user_id
        self.session_id = session_id
    
    @abstractmethod
    async def _get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents for the query"""
        pass
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Sync wrapper for async method"""
        import asyncio
        return asyncio.run(self._get_relevant_documents(query))
