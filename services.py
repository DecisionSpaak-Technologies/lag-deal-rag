import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langsmith import Client
from typing_extensions import TypedDict, List

# Initialize LangSmith client
client = Client()

# Initialize LLM
print("Initializing chat model...")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Initialize embeddings
print("Initializing OpenAI embeddings...")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize vector store
print("Initializing vector store...")
vector_store = Chroma(embedding_function=embeddings)

# Load and split the PDF document
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

# Build the state graph
print("Building state graph...")
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()