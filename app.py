from fastapi import FastAPI, BackgroundTasks, HTTPException
from models import Question, Answer, ProcessingStatus
from services import graph, ensure_vector_store_initialized, get_processing_status, load_and_process_documents
from langsmith import Client
import threading
import logging
import time

app = FastAPI()
client = Client()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flag to track initialization
initialization_started = False

@app.on_event("startup")
async def startup_event():
    global initialization_started
    if not initialization_started:
        logger.info("Starting background data initialization")
        # Run in thread to prevent blocking the server startup
        initialization_started = True
        # Start initialization but don't wait for it to complete
        threading.Thread(target=lambda: ensure_vector_store_initialized(), daemon=True).start()

@app.post("/get_response", response_model=Answer)
async def get_response(question_data: Question):
    try:
        state = {
            "question": question_data.question,
            "session_id": question_data.session_id,
            "context": [],
            "chat_history": "",
            "answer": None
        }
        
        # If documents are still processing, provide status update
        status = get_processing_status()
        if status["status"] == "processing":
            return Answer(
                answer=f"Still loading documents ({status['progress']}/{status['total']}). Please try again in a moment.",
                session_id=question_data.session_id
            )
        elif status["status"] == "error":
            return Answer(
                answer=f"Error loading documents: {status['error']}. Please check server logs.",
                session_id=question_data.session_id
            )
        
        # Execute graph
        result = graph.invoke(state)
        return Answer(
            answer=result["answer"],
            session_id=question_data.session_id
        )
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/processing_status", response_model=ProcessingStatus)
async def processing_status():
    """Get the current status of document processing"""
    status = get_processing_status()
    return ProcessingStatus(
        status=status["status"],
        progress=status["progress"],
        total=status["total"],
        error=status["error"]
    )

@app.post("/reload_documents")
async def reload_documents(background_tasks: BackgroundTasks):
    """Endpoint to force reload documents"""
    from services import vector_store
    
    # Run in background to not block the response
    background_tasks.add_task(lambda: vector_store.delete_collection())
    background_tasks.add_task(lambda: load_and_process_documents(force_reload=True))
    
    return {"status": "Document reload initiated in background"}

@app.post("/initialize_now")
async def initialize_now():
    """Endpoint to manually trigger initialization if it didn't start automatically"""
    global initialization_started
    if not initialization_started:
        initialization_started = True
        threading.Thread(target=lambda: ensure_vector_store_initialized(), daemon=True).start()
        return {"status": "Initialization started"}
    else:
        status = get_processing_status()
        return {"status": f"Initialization already in progress: {status['status']}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)