from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import Chatbot
import uvicorn

app = FastAPI()
chatbot = Chatbot()

# 벡터 스토어 로드
VECTORSTORE_PATH = "vectorstore"
if not chatbot.load_vectorstore(VECTORSTORE_PATH):
    raise Exception("벡터 스토어를 불러올 수 없습니다. 먼저 벡터 스토어를 생성해주세요.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class InitializeRequest(BaseModel):
    texts: list[str]

@app.post("/initialize")
async def initialize(request: InitializeRequest):
    try:
        chatbot.initialize_vectorstore(request.texts)
        return {"status": "success", "message": "벡터 스토어가 초기화되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chatbot.get_response(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=50521)
