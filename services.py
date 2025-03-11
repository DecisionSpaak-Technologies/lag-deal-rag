from langchain_community.document_loaders import UnstructuredAPIFileLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from langchain.memory import VectorStoreRetrieverMemory
import os
import fitz
import dotenv
from datetime import datetime
from models import State
from langchain_core.documents import Document

# Initialize environment
dotenv.load_dotenv()

# Core AI Components
llm = ChatOpenAI(model="gpt-4o-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector Stores
vector_store = Chroma(embedding_function=embeddings)
memory_store = Chroma(
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="conversation_memories"
)

# Document Processing
def load_and_process_documents():
    """Load and process documents with Unstructured API"""
    loader = UnstructuredAPIFileLoader(
        file_path="./data/deal_book.pdf",
        api_key=os.getenv("UNSTRUCTURED_API_KEY"),
        strategy="hi_res",
        ocr_languages=["eng"]
    )
    docs = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    return text_splitter.split_documents(docs)

# Initialize knowledge base
all_splits = load_and_process_documents()
vector_store.add_documents(all_splits)

# Memory System
memory_retriever = memory_store.as_retriever(
    search_kwargs={"k": 3, "filter": {"type": "memory"}}
)
memory = VectorStoreRetrieverMemory(
    retriever=memory_retriever,
    memory_key="chat_history",
    input_key="question"
)

# Graph Components
def retrieve(state: State) -> dict:
    """Retrieval node"""
    docs = vector_store.similarity_search(state["question"])
    return {"context": docs, "question": state["question"]}

def generate(state: State) -> dict:
    """Generation node"""
    response = llm.invoke(
        f"Context: {state['context']}\nQuestion: {state['question']}"
    )
    memory.save_context(
        {"question": state["question"]},
        {"answer": response.content}
    )
    return {"answer": response.content}

# Build workflow graph
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()