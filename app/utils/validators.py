import re
from typing import Optional
from uuid import UUID

def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_phone(phone: str) -> bool:
    """Validate phone number format"""
    # Simple validation - adjust based on your requirements
    pattern = r'^\+?1?\d{9,15}$'
    return bool(re.match(pattern, re.sub(r'[^\d+]', '', phone)))

def validate_session_name(name: str) -> bool:
    """Validate session name"""
    if not name or len(name.strip()) == 0:
        return False
    return len(name.strip()) <= 255

def validate_message_content(content: str) -> bool:
    """Validate message content"""
    if not content or len(content.strip()) == 0:
        return False
    return len(content.strip()) <= 4000

def validate_user_id(user_id: str) -> bool:
    """Validate user ID format"""
    try:
        UUID(user_id)
        return True
    except ValueError:
        return False

def sanitize_input(text: str) -> str:
    """Sanitize user input"""
    if not text:
        return ""
    
    # Remove potential script tags and other dangerous content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = text.strip()
    
    return text