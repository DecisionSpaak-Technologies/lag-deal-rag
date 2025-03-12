from pydantic import BaseModel
from typing import TypedDict, List, Optional, Any, Union

class Question(BaseModel):
    question: str
    session_id: str = "default"

class Answer(BaseModel):
    answer: str
    session_id: str = "default"

class ProcessingStatus(BaseModel):
    status: str  # "idle", "processing", "complete", "error"
    progress: int
    total: int
    error: Optional[str] = None

class State(TypedDict):
    question: str
    context: List[Any]
    chat_history: Any
    answer: Optional[str]
    session_id: str