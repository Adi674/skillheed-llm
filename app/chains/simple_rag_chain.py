# app/chains/simple_rag_chain.py - FIXED VERSION
from typing import Dict, Any, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import logging

from ..config import settings
from ..utils.exceptions import ChainError

logger = logging.getLogger(__name__)

class SimpleRAGChain:
    """Simplified RAG chain that actually works - NO PROBLEMATIC IMPORTS"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self) -> ChatGroq:
        """Initialize Groq LLM"""
        return ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            streaming=settings.groq_streaming
        )
    
    def _format_prompt(self, question: str, context: str = "", recent_messages: str = "") -> str:
        """Create formatted prompt without PromptTemplate"""
        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. Use the provided context to give accurate and relevant responses.

Context:
{context if context else "No additional context available."}

Recent conversation:
{recent_messages if recent_messages else "No recent conversation history."}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return template
    
    async def invoke(self, inputs: Dict[str, Any]) -> str:
        """Process the chain"""
        try:
            # Extract inputs
            question = inputs.get("question", "")
            context = inputs.get("context", "")
            recent_messages = inputs.get("recent_messages", "")
            
            # Format the prompt
            formatted_prompt = self._format_prompt(question, context, recent_messages)
            
            # Call LLM
            response = await self.llm.ainvoke([HumanMessage(content=formatted_prompt)])
            
            # Extract content
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"RAG chain failed: {e}")
            raise ChainError(f"Chain processing failed: {e}")
    
    async def test_chain(self) -> bool:
        """Test if chain is working"""
        try:
            response = await self.invoke({
                "question": "Hello, can you hear me?",
                "context": "Test context",
                "recent_messages": "Test: Hello"
            })
            return len(response) > 0
        except Exception:
            return False

# Global instance
simple_rag_chain = SimpleRAGChain()