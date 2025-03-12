from langchain_community.document_loaders import UnstructuredAPIFileLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import ChatPromptTemplate
import os
import fitz  # PyMuPDF
import dotenv
from models import State
from langchain_core.documents import Document
import logging
import hashlib
import pickle
from pathlib import Path
import time
import threading
import concurrent.futures
import tempfile
import numpy as np
import base64
from PIL import Image
import io
import requests
from typing import List, Dict, Any, Optional, Union

# Initialize environment
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Core AI Components - Using GPT-4-Vision for image support
llm = ChatOpenAI(model="gpt-4o")  # Using a model with vision capabilities
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Vector Stores - FAISS
vector_store_path = "./vector_stores/knowledge_base_faiss"
memory_store_path = "./vector_stores/conversation_memories_faiss"
image_store_path = "./image_store"  # Store for extracted images

# Initialize to None and load lazily
vector_store = None
memory_store = None

# Process status flag
processing_complete = threading.Event()
processing_status = {"status": "idle", "progress": 0, "total": 0, "error": None}

# Ensure necessary directories exist
Path(vector_store_path).parent.mkdir(exist_ok=True)
Path(memory_store_path).parent.mkdir(exist_ok=True)
Path(image_store_path).mkdir(exist_ok=True)

def extract_images_from_pdf(pdf_path: str, output_dir: str) -> List[Dict[str, Any]]:
    """Extract images from PDF and save them with page reference"""
    image_info = []
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Generate a unique name for the image
                image_filename = f"page_{page_num+1}_img_{img_idx+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                # Save the image
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Get the caption if possible (text near the image)
                caption = ""
                try:
                    # Simple approach: extract text from the area around where the image appears
                    rect = page.get_image_bbox(xref)
                    if rect:
                        # Expand the rectangle slightly to capture nearby text
                        expanded_rect = fitz.Rect(rect.x0 - 20, rect.y0 - 20, 
                                                  rect.x1 + 20, rect.y1 + 50)
                        caption = page.get_text("text", clip=expanded_rect)
                        caption = caption.strip()
                except Exception as e:
                    logger.warning(f"Error extracting caption for image on page {page_num+1}: {e}")
                
                # Store image info
                image_info.append({
                    "path": image_path,
                    "page": page_num + 1,
                    "caption": caption,
                    "filename": image_filename
                })
                
                logger.info(f"Extracted image: {image_filename} from page {page_num+1}")
                
        return image_info
    
    except Exception as e:
        logger.error(f"Error extracting images from PDF: {e}")
        return []

def split_pdf_by_pages(input_path, output_dir, pages_per_chunk=10):
    """Split a large PDF into smaller chunks for more efficient processing"""
    try:
        pdf_doc = fitz.open(input_path)
        total_pages = len(pdf_doc)
        chunks = []
        
        logger.info(f"Splitting PDF with {total_pages} pages into chunks of {pages_per_chunk} pages")
        
        for i in range(0, total_pages, pages_per_chunk):
            end_page = min(i + pages_per_chunk - 1, total_pages - 1)
            output_pdf = fitz.open()
            
            for page_num in range(i, min(i + pages_per_chunk, total_pages)):
                output_pdf.insert_pdf(pdf_doc, from_page=page_num, to_page=page_num)
            
            chunk_path = os.path.join(output_dir, f"chunk_{i}_{end_page}.pdf")
            output_pdf.save(chunk_path)
            output_pdf.close()
            
            chunks.append(chunk_path)
            logger.info(f"Saved PDF chunk {len(chunks)}/{(total_pages + pages_per_chunk - 1) // pages_per_chunk}: {chunk_path}")
        
        pdf_doc.close()
        return chunks
    except Exception as e:
        logger.error(f"Error splitting PDF: {str(e)}")
        raise

def process_document_chunk(chunk_path, api_key):
    """Process a single PDF chunk with the Unstructured API"""
    logger.info(f"Processing document chunk: {chunk_path}")
    try:
        loader = UnstructuredAPIFileLoader(
        file_path=chunk_path,
        api_key=api_key,
        strategy="hi_res",
        mode="elements",  # Added mode parameter for text and image extraction
        ocr_languages=["eng"],
        include_image_data=True,  # Important for image extraction
    # Add timeout parameters
        timeout={"connect": 30, "read": 300}  # Longer timeout for large files
)
        return loader.load()
    except Exception as e:
        logger.error(f"Error processing chunk {chunk_path}: {str(e)}")
        return []

def encode_image_to_base64(image_path):
    """Convert an image to base64 encoding for API calls"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def describe_image_with_gpt4(image_path, prompt="Describe what you see in this image in detail"):
    """Use GPT-4 Vision to get a description of an image"""
    try:
        # Encode image
        base64_image = encode_image_to_base64(image_path)
        if not base64_image:
            return "Failed to process image"
        
        # Call GPT-4 Vision with OpenAI client directly
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error describing image with GPT-4: {e}")
        return f"Failed to analyze image: {str(e)}"

def process_images_for_document(file_path):
    """Process all images in a document to create image descriptions"""
    image_docs = []
    
    try:
        # Create a subdirectory for this document's images
        doc_hash = hashlib.md5(file_path.encode()).hexdigest()
        doc_image_dir = os.path.join(image_store_path, doc_hash)
        Path(doc_image_dir).mkdir(exist_ok=True)
        
        # Extract images
        images = extract_images_from_pdf(file_path, doc_image_dir)
        
        # Process each image to get descriptions
        for img in images:
            try:
                # Check if we already have a cached description
                desc_path = f"{img['path']}.description.txt"
                if os.path.exists(desc_path):
                    with open(desc_path, "r") as f:
                        description = f.read()
                else:
                    # Generate a new description
                    prompt = "Describe this image in detail, including any text content visible in the image."
                    if img['caption']:
                        prompt += f" Consider that this image might be related to: {img['caption']}"
                    
                    description = describe_image_with_gpt4(img['path'], prompt)
                    
                    # Cache the description
                    with open(desc_path, "w") as f:
                        f.write(description)
                
                # Create a document for the image
                image_doc = Document(
                    page_content=f"[IMAGE from page {img['page']}]: {description}\n\nCaption or nearby text: {img['caption']}",
                    metadata={
                        "source": file_path,
                        "page": img['page'],
                        "image_path": img['path'],
                        "type": "image"
                    }
                )
                
                image_docs.append(image_doc)
                logger.info(f"Processed image {img['filename']} from page {img['page']}")
                
            except Exception as e:
                logger.error(f"Error processing image {img['path']}: {e}")
        
        return image_docs
    
    except Exception as e:
        logger.error(f"Error in image processing pipeline: {e}")
        return []

def load_and_process_documents(file_path="./data/pdf_data.pdf", force_reload=False, max_workers=3):
    """Load and process documents with caching, chunking for performance, and image extraction"""
    global processing_status
    
    # Create cache directory if it doesn't exist
    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True)
    
    # Create a hash of the file to use as cache key
    file_hash = ""
    file_path_obj = Path(file_path)
    if not file_path_obj.exists():
        logger.error(f"File does not exist: {file_path}")
        processing_status["status"] = "error"
        processing_status["error"] = f"File does not exist: {file_path}"
        processing_complete.set()
        return []
    
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()
    
    cache_file = cache_dir / f"{file_hash}_processed_docs.pkl"
    image_cache_file = cache_dir / f"{file_hash}_image_docs.pkl"
    
    # Check if we have a valid cache
    if cache_file.exists() and image_cache_file.exists() and not force_reload:
        logger.info(f"Loading processed documents from cache: {cache_file}")
        try:
            with open(cache_file, "rb") as f:
                docs = pickle.load(f)
            
            with open(image_cache_file, "rb") as f:
                image_docs = pickle.load(f)
                
            # Combine both document types
            all_docs = docs + image_docs
                
            processing_status["status"] = "complete"
            processing_status["progress"] = 100
            processing_status["total"] = 100
            processing_complete.set()
            return all_docs
        except Exception as e:
            logger.warning(f"Failed to load cache, will reprocess: {str(e)}")
    
    # Update status
    processing_status["status"] = "processing"
    processing_status["progress"] = 0
    processing_status["total"] = 100
    processing_complete.clear()
    
    try:
        # Create a temporary directory for PDF chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            # Split the PDF into smaller chunks
            chunks = split_pdf_by_pages(file_path, temp_dir)
            processing_status["total"] = len(chunks) + 1  # +1 for image processing
            
            # Process each chunk in parallel
            all_docs = []
            api_key = os.getenv("UNSTRUCTURED_API_KEY")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {executor.submit(process_document_chunk, chunk, api_key): chunk for chunk in chunks}
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_chunk)):
                    chunk = future_to_chunk[future]
                    try:
                        chunk_docs = future.result()
                        all_docs.extend(chunk_docs)
                        
                        # Update progress
                        processing_status["progress"] = i + 1
                        logger.info(f"Progress: {i+1}/{len(chunks)} chunks processed")
                    except Exception as e:
                        logger.error(f"Error processing {chunk}: {str(e)}")
            
            # Process images
            logger.info("Processing images from document")
            image_docs = process_images_for_document(file_path)
            
            # Save image docs to cache
            with open(image_cache_file, "wb") as f:
                pickle.dump(image_docs, f)
                
            # Add image docs to all docs
            all_docs.extend(image_docs)
            
            # Split documents - optimized chunk size for FAISS
            logger.info(f"Splitting {len(all_docs)} documents")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=100  # Reduced overlap for faster processing
            )
            splits = text_splitter.split_documents(all_docs)
            
            # Save to cache
            logger.info(f"Saving {len(splits)} document splits to cache")
            with open(cache_file, "wb") as f:
                pickle.dump(splits, f)
            
            # Update status
            processing_status["status"] = "complete"
            processing_status["progress"] = len(chunks) + 1
            processing_complete.set()
            
            return splits
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")
        processing_status["status"] = "error"
        processing_status["error"] = str(e)
        processing_complete.set()
        return []

def get_processing_status():
    """Get the current status of document processing"""
    return processing_status

def ensure_vector_store_initialized(timeout=None):
    """Ensures the vector store is initialized with documents"""
    global vector_store, processing_status
    
    # Initialize vector store directory
    vector_store_dir = Path(vector_store_path).parent
    vector_store_dir.mkdir(exist_ok=True)
    
    # Check if vector store already exists on disk
    if Path(vector_store_path).exists() and os.listdir(vector_store_path):
        try:
            # Load the existing FAISS index
            logger.info("Loading existing FAISS vector store")
            vector_store = FAISS.load_local(vector_store_path, embeddings)
            processing_status["status"] = "complete"
            processing_status["progress"] = 100
            processing_status["total"] = 100
            processing_complete.set()
            return True
        except Exception as e:
            logger.warning(f"Failed to load existing vector store: {str(e)}")
    
    # Start document processing in a separate thread
    logger.info("Initializing vector store with documents")
    thread = threading.Thread(target=lambda: _load_and_add_to_vector_store())
    thread.daemon = True
    thread.start()
    
    # If timeout is specified, wait for processing to complete
    if timeout:
        result = processing_complete.wait(timeout)
        if not result:
            logger.warning(f"Document processing did not complete within {timeout} seconds")
        return result
    
    return True

def _load_and_add_to_vector_store():
    """Helper function to load documents and add to vector store"""
    global vector_store
    try:
        all_splits = load_and_process_documents()
        if all_splits:
            logger.info(f"Creating FAISS index with {len(all_splits)} documents")
            
            # Create FAISS index in batches for memory efficiency
            batch_size = 1000  # Process 1000 documents at a time
            
            for i in range(0, len(all_splits), batch_size):
                batch = all_splits[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(all_splits) + batch_size - 1) // batch_size}")
                
                # For first batch, create the index
                if i == 0:
                    vector_store = FAISS.from_documents(batch, embeddings)
                # For subsequent batches, add to existing index
                else:
                    vector_store.add_documents(batch)
            
            # Save the FAISS index to disk
            vector_store.save_local(vector_store_path)
            logger.info("FAISS vector store created and saved successfully")
        else:
            logger.warning("No documents to add to vector store")
    except Exception as e:
        logger.error(f"Failed to add documents to vector store: {str(e)}")
        processing_status["status"] = "error"
        processing_status["error"] = str(e)
        processing_complete.set()

def get_memory_store():
    """Lazy initialization of memory store"""
    global memory_store
    
    if memory_store is None:
        # Create directory if it doesn't exist
        memory_dir = Path(memory_store_path).parent
        memory_dir.mkdir(exist_ok=True)
        
        # Check if it exists on disk
        if Path(memory_store_path).exists() and os.listdir(memory_store_path):
            try:
                memory_store = FAISS.load_local(
                    memory_store_path,
                    OpenAIEmbeddings(model="text-embedding-3-small")
                )
            except Exception as e:
                logger.warning(f"Failed to load memory store: {str(e)}")
                # Create new if loading fails
                memory_store = FAISS.from_texts(
                    ["Initial memory placeholder"], 
                    OpenAIEmbeddings(model="text-embedding-3-small")
                )
                memory_store.save_local(memory_store_path)
        else:
            # Create new
            memory_store = FAISS.from_texts(
                ["Initial memory placeholder"], 
                OpenAIEmbeddings(model="text-embedding-3-small")
            )
            memory_store.save_local(memory_store_path)
    
    return memory_store

# Memory System - Lazy loaded
def get_memory():
    """Get memory system with lazy loading"""
    memory_retriever = get_memory_store().as_retriever(
        search_kwargs={"k": 3}
    )
    
    return VectorStoreRetrieverMemory(
        retriever=memory_retriever,
        memory_key="chat_history",
        input_key="question",
        return_messages=True
    )

# Define prompt template with image support
PROMPT_TEMPLATE = """
You are a Lagos State Deal book assistant with access to the following context information and chat history.
You are to only answer questions that relates to Lagos State or and Lagos State Deal book. In cases where you do not know the answer to a query, apologise and say you don't have that information.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION: {question}

If any of the context includes image descriptions (marked with [IMAGE]), consider this visual information in your response.
Please provide a detailed answer based on the context and any relevant chat history. 
If you cannot find the answer in the provided context, say so clearly.
"""

prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

# Graph Components
def retrieve(state: State) -> dict:
    """Retrieval node with session context"""
    global vector_store
    
    # Ensure vector store is initialized
    if vector_store is None:
        # Try to load from disk
        try:
            vector_store = FAISS.load_local(vector_store_path, embeddings)
        except Exception as e:
            # If processing is in progress, report status
            if processing_status["status"] == "processing":
                progress = f"{processing_status['progress']}/{processing_status['total']}"
                return {
                    "context": [],
                    "question": state["question"],
                    "chat_history": "",
                    "session_id": state["session_id"],
                    "answer": f"I'm still processing your documents ({progress}). Please try again in a moment."
                }
            # If processing failed, report error
            elif processing_status["status"] == "error":
                return {
                    "context": [],
                    "question": state["question"],
                    "chat_history": "",
                    "session_id": state["session_id"],
                    "answer": f"There was an error processing your documents: {processing_status['error']}"
                }
            else:
                return {
                    "context": [],
                    "question": state["question"],
                    "chat_history": "",
                    "session_id": state["session_id"],
                    "answer": "The document database is not yet available. Please try again later."
                }
    
    # Determine if the question is likely about visual content
    visual_keywords = ["image", "picture", "photo", "diagram", "graph", "chart", "figure", 
                      "illustration", "visual", "shown", "displayed", "looks like", "appears"]
    
    is_visual_query = any(keyword in state["question"].lower() for keyword in visual_keywords)
    
    # Adjust search parameters for visual queries
    k_value = 6 if is_visual_query else 4
    filter_dict = {"type": "image"} if is_visual_query else None
    
    # Get document context with FAISS
    try:
        # First search with standard parameters
        docs = vector_store.similarity_search(
            state["question"],
            k=k_value,
            filter=filter_dict
        )
        
        # For visual queries, ensure we have both text and image content
        if is_visual_query and filter_dict:
            # Check if we need to add regular text content
            if len(docs) < k_value:
                text_docs = vector_store.similarity_search(
                    state["question"],
                    k=k_value - len(docs),
                    filter=None
                )
                docs.extend(text_docs)
            
        # If no visual content was found but it seems to be a visual query, add a note
        if is_visual_query and not any("IMAGE" in doc.page_content for doc in docs):
            logger.info("Visual query detected but no image content found in results")
    
    except Exception as e:
        logger.error(f"Error in similarity search: {str(e)}")
        docs = []
    
    # Get memory context if session_id provided
    chat_history = ""
    if state["session_id"] != "default":
        try:
            memory = get_memory()
            memory_result = memory.load_memory_variables({"question": state["question"]})
            chat_history = memory_result.get("chat_history", "")
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
    
    return {
        "context": docs, 
        "question": state["question"],
        "chat_history": chat_history,
        "session_id": state["session_id"]
    }

def generate(state: State) -> dict:
    """Generation node with improved prompt"""
    # If we already have an answer (e.g., from error handling), return it
    if state.get("answer"):
        return {"answer": state["answer"]}
        
    # Format context for prompt
    context_text = "\n\n".join([doc.page_content for doc in state["context"]]) if state["context"] else "No relevant context found."
    
    # Check if we have image content
    has_image_content = any("[IMAGE" in doc.page_content for doc in state["context"])
    
    # Generate response using the prompt template
    try:
        response = llm.invoke(
            prompt.format(
                context=context_text,
                chat_history=state["chat_history"],
                question=state["question"]
            )
        )
        
        answer = response.content
        
        # Add a note if the question seems visual but we don't have image content
        visual_keywords = ["image", "picture", "photo", "diagram", "graph", "chart", "figure"]
        if any(keyword in state["question"].lower() for keyword in visual_keywords) and not has_image_content:
            answer += "\n\n(Note: I wasn't able to find any relevant images or visual content in the document that matches your query.)"
            
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        answer = f"I encountered an error while generating a response. Please try again."
    
    # Save to memory if using a session
    if state["session_id"] != "default":
        try:
            memory = get_memory()
            memory.save_context(
                {"question": state["question"]},
                {"output": answer}
            )
            
            # Periodically save memory to disk (every 10 interactions based on memory size)
            memory_store = get_memory_store()
            if len(memory_store.index_to_docstore_id) % 10 == 0:
                memory_store.save_local(memory_store_path)
                logger.info("Memory store saved to disk")
                
        except Exception as e:
            logger.error(f"Error saving to memory: {str(e)}")
    
    return {"answer": answer}

# Build workflow graph
graph_builder = StateGraph(State)
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "generate")
graph = graph_builder.compile()