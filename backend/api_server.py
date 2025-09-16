import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import MathTutorAgent, feedbackAgent
from vdb_updater import get_updater
from dotenv import load_dotenv

load_dotenv()

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "groq")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")

app = FastAPI(title="MathTutor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent_instance = MathTutorAgent(model_provider=MODEL_PROVIDER, model_name=MODEL_NAME)
feedback_instance = feedbackAgent(model_provider=MODEL_PROVIDER, model_name=MODEL_NAME)

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    error: str | None = None

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    feedback: str

class FeedbackResponse(BaseModel):
    improved_answer: str
    error: str | None = None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    try:
        answer = await agent_instance.get_response(req.question)
        updater = get_updater()
        success = updater.add_qa_pair(req.question, answer)
        if not success:
            print("Failed to store Q/A pair in vector DB")
        else:
            print("Stored Q/A pair in vector DB")
        return AskResponse(answer=answer)
    except Exception as e:
        return AskResponse(answer="", error=str(e))

@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    if not (req.question.strip() and req.answer.strip() and req.feedback.strip()):
        raise HTTPException(status_code=400, detail="question, answer and feedback required")
    try:
        improved = feedback_instance.get_feedback_answer(req.question, req.answer, req.feedback)
        return FeedbackResponse(improved_answer=improved)
    except Exception as e:
        return FeedbackResponse(improved_answer="", error=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010, reload=False)
