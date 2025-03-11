from pydantic import BaseModel
from typing import TypedDict, List
from langchain_core.documents import Document

class Question(BaseModel):
    question: str
    session_id: str = "default"

class Answer(BaseModel):
    answer: str
    session_id: str = "default"

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    session_id: str