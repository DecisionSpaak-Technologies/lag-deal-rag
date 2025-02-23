from pydantic import BaseModel
from typing import List
from typing_extensions import TypedDict
from langchain_core.documents import Document

# Pydantic models for API request/response
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

# State type for the application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str