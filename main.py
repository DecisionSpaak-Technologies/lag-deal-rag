from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import os
import dotenv

from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = dotenv.get_key('.env', 'LANGCHAIN_API_KEY')
os.environ["OPENAI_API_KEY"] = dotenv.get_key('.env', 'OPENAI_API_KEY')


print("Initializing chat model...")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

print("Initializing OpenAI embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

print("Initializing vector store...")
vector_store = Chroma(embedding_function=embeddings)

# Load and chunk contents of the pdf
print("Loading PDF document...")
loader = PyPDFLoader("./data/deal_book.pdf")
print('Document Loaded Successfully')

# Load pages synchronously
print("Loading pages from PDF...")
docs = loader.load()
print('Page loaded successfully')

print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
print("Indexing chunks into vector store...")
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
print("Pulling RAG prompt from hub...")
prompt = hub.pull("rlm/rag-prompt")

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    print("Retrieving relevant documents...")
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    print("Generating answer...")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Compile application and test
print("Building state graph...")
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

app = FastAPI()

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@app.post("/get_response", response_model=Answer)
async def get_response(question_data: Question):

    # Call the graph.invoke function with the provided question
    response = graph.invoke({"question": question_data.question})
    
    # Extract and return the answer
    return {"answer": response["answer"]}

if __name__ == "__main__":
    print("Starting FastAPI server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)