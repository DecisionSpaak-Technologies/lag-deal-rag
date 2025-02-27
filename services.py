import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader

from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from langsmith import Client
from typing_extensions import TypedDict, List

from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import Document
from datetime import datetime  

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
loader = PyMuPDFLoader("./data/deal_book.pdf")
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

# ===================== Memory Setup =====================
# Separate vector store for memories
memory_store = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="conversation_memories"
)

# Memory retriever (last 3 relevant interactions)
memory_retriever = memory_store.as_retriever(
    search_kwargs={
        "k": 3,
        "filter": {
            "$and": [
                {"session_id": {"$eq": "default"}},  # Use $eq operator
                {"type": {"$eq": "memory"}}
            ]
        }
    }
)

# Initialize memory system
memory = VectorStoreRetrieverMemory(
    retriever=memory_retriever,
    memory_key="chat_history",
    input_key="question",
    session_key="session_id"
)

# ===================== Modified Pipeline =====================
# ... (previous imports)

def retrieve(state: State) -> dict:
    session_id = state.get("session_id", "default")
    main_docs = vector_store.similarity_search(state["question"])
    
    # Update filter syntax
    memory_docs = memory_store.similarity_search(
        state["question"],
        k=3,
        filter={
            "$and": [
                {"session_id": {"$eq": session_id}},
                {"type": {"$eq": "memory"}}
            ]
        }
    )
    
    return {
        "context": main_docs + memory_docs,
        "session_id": session_id
    }

def generate(state: State) -> dict:
    # Ensure session_id exists
    session_id = state.get("session_id", "default")
    
    # Generate answer
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    response = llm.invoke(prompt.format(
        question=state["question"],
        context=docs_content
    ))
    
    # Save to memory
    memory_store.add_documents([Document(
        page_content=f"Q: {state['question']}\nA: {response.content}",
        metadata={
            "session_id": session_id,
            "type": "memory",
            "timestamp": datetime.now().isoformat()
        }
    )])
    
    return {
        "answer": response.content,
        "session_id": session_id  # Pass through session_id
    }

# Build the graph with explicit state handling
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_edge(START, "retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()