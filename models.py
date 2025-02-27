from pydantic import BaseModel
from typing import List, TypedDict, Optional
from langchain_core.documents import Document

# Request/Response Schemas
class Question(BaseModel):
    question: str
    session_id: str = "default"  # For multi-user support

class Answer(BaseModel):
    answer: str
    session_id: str = "default"

# State with Memory Context
class State(TypedDict):
    question: str
    session_id: str  # Required field
    context: List[Document]
    answer: Optional[str]  # Allow None for initial state