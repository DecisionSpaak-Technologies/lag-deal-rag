from fastapi import FastAPI
from models import Question, Answer
from services import graph
from langsmith import traceable, Client
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.chat_models import init_chat_model
from langchain.evaluation import load_evaluator

app = FastAPI()
client = Client()
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

@traceable
@app.post("/get_response", response_model=Answer)
async def get_response(question_data: Question):
    response = graph.invoke({"question": question_data.question})
    return {"answer": response["answer"]}

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
# dataset = client.create_dataset(
#     dataset_name,
#     description="Evaluation dataset for the Lagos State Government deal book",
# )

# ===================== Fixed Evaluation Section =====================
# 1. Create evaluators using load_evaluator (not dictionaries)
# criteria_evaluators = [
#     load_evaluator("criteria", criteria="conciseness"),
#     load_evaluator("criteria", criteria="relevance"),
#     load_evaluator("criteria", criteria="correctness"),
#     load_evaluator("criteria", criteria="helpfulness"),
# ]

# 2. Define a proper chain factory
def chain_factory(example: dict):
    return graph.invoke(example["inputs"])

# 3. Configure RunEvalConfig correctly
# eval_config = RunEvalConfig(
#     evaluators=criteria_evaluators,
#     input_key="question",  # Match your dataset's input key
#     prediction_key="answer",  # Match your chain's output key
# )

# 4. Run evaluation with the chain factory
print("Running evaluation on dataset...")
# run_on_dataset(
#     client=client,
#     dataset_name=dataset_name,
#     llm_or_chain_factory=chain_factory,  # Use chain factory, not raw LLM
#     evaluation=eval_config,
# )

# ================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)