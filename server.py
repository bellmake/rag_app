from fastapi import FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes

from chains import ChatChain, TopicChain, LLM, Translator, RagChatChain
# from rag import RagChain  # 더 이상 사용하지 않으므로 주석 처리
from rag import get_welcome_message

from dotenv import load_dotenv
import glob
from pathlib import Path

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

def get_rag_pdfs():
    # (기존 로직 유지)
    current = Path(__file__).resolve()
    while not (current / "data").exists():
        if current.parent == current:
            raise RuntimeError("프로젝트 루트에서 'data' 폴더를 찾을 수 없습니다.")
        current = current.parent
    pdf_dir = current / "data" / "minimum"
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    return [str(p) for p in pdfs]

# 루트 경로: 환영 화면만 보여주도록 변경
@app.get("/", response_class=HTMLResponse)
async def root_with_welcome():
    welcome = get_welcome_message()
    return f"""
    <html>
        <head><meta charset='utf-8'><title>선율RAG 환영</title></head>
        <body style='font-family:sans-serif;text-align:center;margin-top:10em;'>
            <h2>{welcome}</h2>
            <a href='/chat/playground' style='padding:1em 2em;background:#0078d4;color:white;border-radius:8px;text-decoration:none;'>채팅 시작하기</a>
        </body>
    </html>
    """

# 나머지 엔드포인트 등록 (변경 없음)
add_routes(app, Translator().create(), path="/translate")
add_routes(app, LLM().create(), path="/llm")
add_routes(app, TopicChain().create(), path="/topic")
add_routes(app, RagChatChain(file_paths=get_rag_pdfs()).create(), path="/rag")

# 챗 인터페이스
class InputChat(BaseModel):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(...)

add_routes(
    app,
    RagChatChain(file_paths=get_rag_pdfs()).create().with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

# 챗 전용 환영 메시지
@app.get("/chat/welcome")
async def chat_welcome():
    return {"message": get_welcome_message()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)