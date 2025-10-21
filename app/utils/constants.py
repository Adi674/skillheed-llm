"""Constants used throughout the application"""

# API Response Messages
class ResponseMessages:
    SUCCESS = "Operation completed successfully"
    ERROR = "An error occurred"
    NOT_FOUND = "Resource not found"
    UNAUTHORIZED = "Unauthorized access"
    INVALID_INPUT = "Invalid input provided"
    RATE_LIMITED = "Rate limit exceeded"
    SERVICE_UNAVAILABLE = "Service temporarily unavailable"

# Vector Search Constants
class VectorConstants:
    DEFAULT_TOP_K = 5
    MAX_TOP_K = 20
    MIN_SIMILARITY_SCORE = 0.3
    DEFAULT_SIMILARITY_THRESHOLD = 0.7

# Memory Management Constants
class MemoryConstants:
    DEFAULT_CONTEXT_WINDOW = 10
    MAX_CONTEXT_WINDOW = 50
    DEFAULT_SUMMARY_THRESHOLD = 20
    MAX_SUMMARY_THRESHOLD = 100
    DEFAULT_MAX_TOKENS = 4000

# Session Constants
class SessionConstants:
    DEFAULT_SESSION_NAME = "New Chat"
    MAX_SESSION_NAME_LENGTH = 255
    MAX_SESSIONS_PER_USER = 1000

# Message Constants
class MessageConstants:
    MAX_MESSAGE_LENGTH = 4000
    MIN_MESSAGE_LENGTH = 1
    MAX_MESSAGES_PER_SESSION = 10000