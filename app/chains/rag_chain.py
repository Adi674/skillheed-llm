from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import logging

from ..config import settings, LLAMA_MODEL_INFO
from ..utils.exceptions import ChainError

logger = logging.getLogger(__name__)

class RAGChain:
    """Main RAG chain for the chatbot using LangChain and Groq Llama 3.3 70B Versatile"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.prompt = self._create_prompt_template()
    
    def _initialize_llm(self) -> ChatGroq:
        """Initialize Groq LLM with Llama 3.3 70B Versatile"""
        try:
            llm = ChatGroq(
                groq_api_key=settings.groq_api_key,
                model_name=settings.groq_model,  # llama-3.3-70b-versatile
                temperature=settings.groq_temperature,
                max_tokens=settings.groq_max_tokens,
                streaming=settings.groq_streaming
            )
            logger.info(f"Initialized Groq LLM: {settings.groq_model}")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize Groq Llama 3.3 70B: {e}")
            raise ChainError(f"LLM initialization failed: {e}")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create optimized prompt template for Llama 3.3 70B Versatile"""
        # Llama 3.3 chat template format
        template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{settings.llama_system_prompt}

You have access to conversation context and relevant information to help answer questions accurately. Use this context to provide helpful, relevant responses.

Context Information:
{{context}}

Recent Conversation:
{{recent_messages}}

Guidelines:
- Use the provided context to inform your responses
- If context is relevant, reference it naturally in your answer
- If you don't have enough information, say so clearly
- Keep responses concise but comprehensive
- Maintain conversation flow and context
- Be helpful and conversational

<|eot_id|><|start_header_id|>user<|end_header_id|>

{{question}}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return PromptTemplate(
            input_variables=["context", "recent_messages", "question"],
            template=template
        )
    
    async def generate_response(
        self,
        question: str,
        context: str = "",
        recent_messages: str = ""
    ) -> str:
        """Generate response using Llama 3.3 70B Versatile - FIXED"""
        try:
            # Format the prompt with Llama 3.3 chat template
            formatted_prompt = self.prompt.format(
                context=context or "No additional context available.",
                recent_messages=recent_messages or "No recent conversation history.",
                question=question
            )
            
            # Log for debugging
            logger.info(f"Using Groq model: {settings.groq_model}")
            logger.info(f"Formatted prompt length: {len(formatted_prompt)} chars")
            logger.info(f"System prompt: {settings.llama_system_prompt[:100]}...")
            
            # FIXED: Actually invoke the LLM
            try:
                # Create a human message for the LLM
                message = HumanMessage(content=formatted_prompt)
                
                # Call the LLM
                logger.info("Calling Groq LLM...")
                response = await self.llm.ainvoke([message])
                
                # Extract content from response
                if hasattr(response, 'content'):
                    content = response.content.strip()
                else:
                    content = str(response).strip()
                
                logger.info(f"Groq response received: {content[:100]}...")
                return content
                
            except Exception as llm_error:
                logger.error(f"Groq LLM call failed: {llm_error}")
                # Fallback response with error details
                return f"I apologize, but I'm having trouble connecting to the AI service right now. Error: {str(llm_error)}. Please try again in a moment."
            
        except Exception as e:
            logger.error(f"Failed to generate response with Llama 3.3: {e}")
            raise ChainError(f"Response generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Llama 3.3 70B model"""
        return {
            **LLAMA_MODEL_INFO,
            "current_settings": {
                "temperature": settings.groq_temperature,
                "max_tokens": settings.groq_max_tokens,
                "streaming": settings.groq_streaming,
                "model": settings.groq_model
            }
        }
    
    async def test_llm_connection(self) -> bool:
        """Test if LLM connection is working"""
        try:
            test_response = await self.generate_response("Hello, can you hear me?")
            return len(test_response) > 0
        except Exception:
            return False