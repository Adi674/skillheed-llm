import re
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

def generate_uuid() -> str:
    """Generate a new UUID string"""
    return str(uuid.uuid4())

def validate_uuid(uuid_string: str) -> bool:
    """Validate if string is a valid UUID"""
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False

def clean_text(text: str) -> str:
    """Clean and normalize text for processing"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    return text

def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."

def format_timestamp(dt: datetime) -> str:
    """Format datetime to ISO string"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def parse_timestamp(timestamp_str: str) -> datetime:
    """Parse ISO timestamp string to datetime"""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        return datetime.utcnow()

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text (simple implementation)"""
    # Simple keyword extraction - in production, use proper NLP
    words = clean_text(text.lower()).split()
    
    # Filter out common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return keywords[:max_keywords]

def calculate_similarity_score(score: float) -> str:
    """Convert similarity score to human readable format"""
    if score >= 0.9:
        return "very_high"
    elif score >= 0.7:
        return "high"
    elif score >= 0.5:
        return "medium"
    elif score >= 0.3:
        return "low"
    else:
        return "very_low"

def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split list into batches of specified size"""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        result.update(d)
    return result

def safe_get(dictionary: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get value from dictionary with nested key support"""
    keys = key.split('.')
    value = dictionary
    
    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default

def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """Estimate reading time in minutes"""
    word_count = len(text.split())
    return max(1, word_count // words_per_minute)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, limit: int, window_seconds: int) -> bool:
        """Check if request is allowed under rate limit"""
        now = datetime.utcnow().timestamp()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests outside the window
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if now - req_time < window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[key]) < limit:
            self.requests[key].append(now)
            return True
        
        return False