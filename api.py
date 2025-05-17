# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from backend.rag_agent import TherapyRAGAgent
import uvicorn

app = FastAPI()
agent = TherapyRAGAgent()

class ChatRequest(BaseModel):
    message: str
    chat_history: list = []

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = agent.chat(request.message, request.chat_history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-documents")
async def load_documents(directory: str):
    try:
        agent.load_documents(directory)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)