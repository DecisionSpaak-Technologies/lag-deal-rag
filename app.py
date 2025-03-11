from fastapi import FastAPI
from models import Question, Answer
from services import graph, memory
from langsmith import Client

app = FastAPI()
client = Client()

@app.post("/get_response", response_model=Answer)
async def get_response(question_data: Question):
    state = {
        "question": question_data.question,
        "session_id": question_data.session_id,
        "context": [],
        "answer": None
    }
    
    result = graph.invoke(state)
    return Answer(
        answer=result["answer"],
        session_id=question_data.session_id
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)