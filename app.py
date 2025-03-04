from fastapi import FastAPI
from models import Question, Answer
from services import graph, memory_store
from langsmith import traceable, Client
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.chat_models import init_chat_model
from langchain.evaluation import load_evaluator
from datetime import datetime

app = FastAPI()
client = Client()
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

@traceable
@app.post("/get_response", response_model=Answer)
async def get_response(question_data: Question):
    # Initialize state with all required fields
    initial_state = {
        "question": question_data.question,
        "session_id": question_data.session_id,
        "context": [],
        "image_context": [],  # NEW FIELD
        "answer": None
    }
    
    response = graph.invoke(initial_state)
    return response

# Define evaluation examples (expand as needed)
example_inputs = [
    "Tell me about the current power situation in Lagos",
    "What is the Purple Line Rail Project",
    "What are the free zone benefits",
    "What are the projects Lagos has for tourism",
    "What are the projects Lagos has for agriculture",
    "What is the population of Lagos State",
]

# Create an evaluation dataset on LangSmith
dataset_name = "LAG_DEAL_RAG_EVAL_1"


# Define a proper chain factory
def chain_factory(example: dict):
    return graph.invoke(example["inputs"])

# print("Running evaluation on dataset...")


# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)