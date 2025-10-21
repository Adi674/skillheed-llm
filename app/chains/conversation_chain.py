from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConversationChain:
    """Chain for managing conversation flow and context with Llama 3.3 70B"""
    
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
    
    def format_recent_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Format recent messages for Llama 3.3 context"""
        if not messages:
            return "No recent conversation."
        
        formatted = []
        for msg in messages[-10:]:  # Last 10 messages
            role = msg.get('role', '').lower()
            content = msg.get('content', '')
            
            # Format according to Llama 3.3 chat template
            if role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted)
    
    def format_context(self, vector_results: Dict[str, List[Dict]], summaries: List[Dict]) -> str:
        """Format retrieved context optimized for Llama 3.3 understanding"""
        context_parts = []
        
        # Add conversation summaries first (most important context)
        if summaries:
            context_parts.append("=== Previous Conversation Summaries ===")
            for summary in summaries[:2]:
                summary_text = summary.get('summary_text', '')
                message_count = summary.get('message_count', 0)
                context_parts.append(f"Previous conversation ({message_count} messages): {summary_text}")
        
        # Add relevant messages from vector search
        if vector_results.get('messages'):
            context_parts.append("\n=== Relevant Past Messages ===")
            for result in vector_results['messages'][:3]:
                metadata = result.get('metadata', {})
                role = metadata.get('role', 'unknown')
                timestamp = metadata.get('timestamp', '')
                session_name = metadata.get('session_name', 'conversation')
                score = result.get('score', 0)
                
                context_parts.append(
                    f"From {session_name} ({timestamp[:10]}): {role} message (relevance: {score:.2f})"
                )
        
        # Add vector summaries if available
        if vector_results.get('summaries'):
            context_parts.append("\n=== Related Conversations ===")
            for result in vector_results['summaries'][:2]:
                metadata = result.get('metadata', {})
                message_count = metadata.get('message_count', 0)
                session_name = metadata.get('session_name', 'conversation')
                score = result.get('score', 0)
                
                context_parts.append(
                    f"Related {session_name}: {message_count} messages (relevance: {score:.2f})"
                )
        
        return "\n".join(context_parts) if context_parts else "No additional context available."
    
    async def process_conversation(
        self,
        question: str,
        recent_messages: List[Dict[str, Any]],
        vector_results: Dict[str, List[Dict]],
        summaries: List[Dict[str, Any]]
    ) -> str:
        """Process conversation with optimized context for Llama 3.3 70B - FIXED"""
        try:
            # Format context and recent messages
            context = self.format_context(vector_results, summaries)
            recent_formatted = self.format_recent_messages(recent_messages)
            
            # Log context for debugging
            logger.info(f"Context: {context[:200]}...")
            logger.info(f"Recent messages: {recent_formatted[:200]}...")
            logger.info(f"Question: {question}")
            
            # FIXED: Actually call the RAG chain instead of placeholder
            response = await self.rag_chain.generate_response(
                question=question,
                context=context,
                recent_messages=recent_formatted
            )
            
            logger.info(f"Generated response: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"Failed to process conversation with Llama 3.3: {e}")
            return f"I apologize, but I encountered an error while processing your message: {str(e)}. Please try again."