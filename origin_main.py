import os
import dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing_extensions import TypedDict, List

# Import LangChain and LangSmith components
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph

from langsmith import traceable, Client
from langchain.smith import RunEvalConfig, run_on_dataset
# Import BaseCache and rebuild the RunEvalConfig model
# from langchain.schema.cache import BaseCache
# RunEvalConfig.model_rebuild(BaseCache=BaseCache)

# Set up environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = dotenv.get_key('.env', 'LANGCHAIN_API_KEY')
os.environ["OPENAI_API_KEY"] = dotenv.get_key('.env', 'OPENAI_API_KEY')
os.environ["LANGSMITH_ENDPOINT"] = 'https://api.smith.langchain.com'
os.environ["LANGSMITH_PROJECT"] = dotenv.get_key('.env', 'LANGSMITH_PROJECT')

# Initialize LangSmith client, LLM, embeddings, and vector store
client = Client()

print("Initializing chat model...")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

print("Initializing OpenAI embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

print("Initializing vector store...")
vector_store = Chroma(embedding_function=embeddings)

# Load the PDF document and split it into chunks
print("Loading PDF document...")
loader = PyPDFLoader("./data/deal_book.pdf")
docs = loader.load()
print("Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index the chunks into the vector store
print("Indexing chunks into vector store...")
_ = vector_store.add_documents(documents=all_splits)

# Load the prompt for question-answering from the hub
print("Pulling RAG prompt from hub...")
prompt = hub.pull("rlm/rag-prompt")

# Define the state type for our application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define pipeline steps: retrieval and generation
def retrieve(state: State) -> dict:
    print("Retrieving relevant documents...")
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State) -> dict:
    print("Generating answer...")
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Build the state graph from the defined steps
print("Building state graph...")
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Define the FastAPI app and routes
app = FastAPI()

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

@traceable
@app.post("/get_response", response_model=Answer)
async def get_response(question_data: Question):
    response = graph.invoke({"question": question_data.question})
    return {"answer": response["answer"]}

# ===================== Evaluation Section =====================
# Define a chain factory that runs the full pipeline using the state graph.
# def chain_factory(example: dict) -> dict:
#     # Each example is expected to have an "inputs" dict with a "question" key.
#     return graph.invoke(example["inputs"])

if __name__ == "__main__":
    print("Starting FastAPI server...")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)