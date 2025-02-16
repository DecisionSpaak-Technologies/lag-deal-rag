from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import os

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict



os.environ["USER_AGENT"] = "MyApp/1.0 (myapp@example.com)"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ''
os.environ["OPENAI_API_KEY"] = ""

llm = init_chat_model("gpt-4o-mini", model_provider="openai")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(embedding_function=embeddings)

# Load and chunk contents of the pdf
loader = PyPDFLoader("./data/pdf_data.pdf")

# Load pages synchronously
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    relevant_chunks: List[str]  # New field to store relevant chunks

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    
    
    relevant_chunks = [doc.page_content for doc in retrieved_docs]
    return {"context": retrieved_docs, "relevant_chunks": relevant_chunks}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content, "relevant_chunks": state["relevant_chunks"]}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

app = FastAPI()

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    relevant_chunks: List[str]  # New field to return relevant chunks

@app.post("/get_response", response_model=Answer)
async def get_response(question_data: Question):
    try:
        # Call the graph.invoke function with the provided question
        response = graph.invoke({"question": question_data.question})
        
        # Extract and return the answer and relevant chunks
        return {
            "answer": response["answer"],
            "relevant_chunks": response["relevant_chunks"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)