from fastapi import FastAPI
from models import Question, Answer
from services import graph
from langsmith import traceable

app = FastAPI()

@traceable
@app.post("/get_response", response_model=Answer)
async def get_response(question_data: Question):
    response = graph.invoke({"question": question_data.question})
    return {"answer": response["answer"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)