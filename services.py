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

from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate

from langchain_core.messages import HumanMessage
import base64

from PIL import Image
import io
import fitz  # PyMuPDF

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

def is_visual_question(question: str) -> bool:
    visual_keywords = {"image", "picture", "diagram", "chart", "graph", "photo", "visual"}
    return any(keyword in question.lower() for keyword in visual_keywords)


rag_prompt = PromptTemplate.from_template("""
system_message = You are a Helpful assistant bot and a document expert with access to both text content and image descriptions. Follow these rules:
1. Only When asked about visual content, ALWAYS check image context.
2. You can only answer questions relating to Lagos State.
5. If you don't know the answer to a question, say something like unfortunately, do not have that information.
                                                                                   
You are an expert assistant analyzing documents with both text and images. 
Use the following pieces of context to answer the question at the end.
If the question asks about images or visual content, use the image descriptions provided, if not, do not mention anything aboutfailing to download image Also, you are The Lagos State Deal Book Chatbot. If the user does not ask anything about an image, you can ignore the image context, and no need then to say a single thing about an image.

**Text Context:**
{text_context}

**Image Context:**
{image_context}

**Chat History:**
{chat_history}

Question: {question}
Answer in detail, including image descriptions when relevant:If image descriptions show errors, mention: "Could not retrieve image description" """)


examples = [
    {
        "question": "What does the infrastructure diagram look like?",
        "text_context": "The infrastructure plan outlines three phases of development",
        "image_context": "Image from page 12:\nA detailed diagram showing three phases of construction...",
        "answer": "The infrastructure diagram (page 12) shows three phases of construction..."
    },
    {
        "question": "Describe the chart about investment growth",
        "text_context": "Annual reports mention increasing investments",
        "image_context": "Image from page 8:\nA line chart displaying 20% year-over-year growth...",
        "answer": "The investment growth chart (page 8) depicts a steady 20% year-over-year increase..."
    }
]

final_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate.from_template(
        "Question: {question}\nText Context: {text_context}\nImage Context: {image_context}\nAnswer: {answer}"
    ),
    prefix=rag_prompt.template,
    suffix="Question: {question}\nAnswer:",
    input_variables=["text_context", "image_context", "question", "chat_history"]
)

# Load the prompt for question-answering from the hub
print("Pulling RAG prompt from hub...")
prompt = final_prompt


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

def extract_images_from_pdf(pdf_path, output_dir="data/images"):
    os.makedirs(output_dir, exist_ok=True)
    image_metadata = []
    
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Save image
            # In extract_images_from_pdf
            image_path = os.path.abspath(os.path.join(output_dir, f"page_{page_num+1}_img_{img_index+1}.png"))
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            # Get surrounding text
            text = page.get_text("text")  # Or use bounding box coordinates
            image_metadata.append({
                "path": image_path,
                "page": page_num+1,
                "text": text[:500]  # Store first 500 chars for context
            })
    
    return image_metadata

# Add to your PDF loading process
print("Extracting images from PDF...")
image_metadata = extract_images_from_pdf("./data/deal_book.pdf")

class ImageDescriber:
    def __init__(self):
        self.vision_model = init_chat_model(model="gpt-4o")
        
    def describe_image(self, image_path):
        

        # Read and encode the image
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            return f"Error loading image: {str(e)}"

        # Create proper base64 URL
        image_url = f"data:image/png;base64,{base64_image}"
        
        # Create message
        image_message = {
            "type": "image_url",
            "image_url": {
                "url": image_url,
                "detail": "auto"  # Can be "low", "high", or "auto"
            }
        }
        
        try:
            response = self.vision_model.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": "Describe this image in detail for someone who can't see it. Include text content if present."},
                    image_message
                ])
            ])
            return response.content
        except Exception as e:
            return f"Error generating description: {str(e)}"
    def __init__(self):
        self.vision_model = init_chat_model(model="gpt-4o")
        
    def describe_image(self, image_path):
        from langchain_core.messages import HumanMessage
        
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"file://{image_path}"}
        }
        
        response = self.vision_model.invoke([
            HumanMessage(content=[
                {"type": "text", "text": "Describe this image in detail for someone who can't see it. Include text content if present."},
                image_message
            ])
        ])
        
        return response.content

image_describer = ImageDescriber()

# Define the state type for our application
class State(TypedDict):
    question: str
    context: List[Document]
    image_context: List[Document]
    answer: str
    session_id: str 

print("Indexing image metadata...")
image_docs = [
    Document(
        page_content=img["text"],
        metadata={
            "type": "image_metadata",
            "image_path": img["path"],
            "page": img["page"]
        }
    ) for img in image_metadata
]
_ = vector_store.add_documents(image_docs)

def retrieve(state: State) -> dict:
    session_id = state.get("session_id", "default")
    question = state["question"]
    is_visual = is_visual_question(question)
    
    # Retrieve main text documents
    main_docs = vector_store.similarity_search(question)
    
    # Retrieve image documents only for visual questions
    image_docs = []
    if is_visual:
        image_docs = vector_store.similarity_search(
            question,
            k=2,
            filter={"type": {"$eq": "image_metadata"}}
        )
    
    # Retrieve memory documents
    memory_docs = memory_store.similarity_search(
        question,
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
        "image_context": image_docs,
        "is_visual": is_visual,  # Pass visual flag to next stage
        "session_id": session_id
    }

def generate(state: State) -> dict:
    session_id = state.get("session_id", "default")
    question = state["question"]
    is_visual = state.get("is_visual", False)
    
    # Process images only for visual questions
    image_descriptions = []
    if is_visual:
        for doc in state.get("image_context", []):
            if "image_path" in doc.metadata:
                try:
                    description = image_describer.describe_image(doc.metadata["image_path"])
                    image_descriptions.append(
                        f"Image from page {doc.metadata['page']}:\n{description}"
                    )
                except Exception as e:
                    print(f"Error describing image: {str(e)}")
    
    # Prepare context sections
    text_context = "\n\n".join(doc.page_content for doc in state["context"])
    image_context = "\n\n".join(image_descriptions) if image_descriptions else ""
    
    # Dynamic prompt construction
    prompt_sections = [
        "**Text Context:**",
        text_context,
        "**Chat History:**",
        memory.load_memory_variables({"question": question})["chat_history"],
        "Question: {question}"
    ]
    
    # Add image section only for visual questions
    if is_visual:
        prompt_sections.insert(2, "**Image Context:**\n" + (image_context if image_context else "No relevant images"))
    
    response = llm.invoke("\n\n".join(prompt_sections).format(question=question))
    
    # Save to memory
    memory_store.add_documents([Document(
        page_content=f"Q: {question}\nA: {response.content}",
        metadata={
            "session_id": session_id,
            "type": "memory",
            "timestamp": datetime.now().isoformat()
        }
    )])
    
    return {"answer": response.content, "session_id": session_id}

# Build the graph with explicit state handling
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()
