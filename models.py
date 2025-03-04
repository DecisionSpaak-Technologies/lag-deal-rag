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

# State with Memory Context# UPDATE STATE CLASS DEFINITION
class State(TypedDict):
    question: str
    context: List[Document]
    image_context: List[Document]
    answer: str
    session_id: str
    is_visual: bool