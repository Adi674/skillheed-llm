from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class SessionStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

class VectorType(str, Enum):
    MESSAGE = "message"
    SUMMARY = "summary"
    KNOWLEDGE = "knowledge"
