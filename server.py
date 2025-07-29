from fastapi import FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langserve import add_routes

from chains import ChatChain, TopicChain, LLM, Translator, RagChatChain
from rag import RagChain

from dotenv import load_dotenv
import glob
from pathlib import Path

load_dotenv()

# FastAPI 애플리케이션 객체 초기화
app = FastAPI()

# CORS 미들웨어 설정
# 외부 도메인에서의 API 접근을 위한 보안 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# R24-11 AP 폴더 내 모든 PDF 파일 경로 수집
def get_rag_pdfs():
    # 1) server.py가 어디에 있든, 프로젝트 루트(즉 data 폴더가 있는 곳)를 찾는다.
    current = Path(__file__).resolve()
    while not (current / "data").exists():
        if current.parent == current:
            # 루트까지 올라갔는데도 data 폴더가 없으면 에러
            raise RuntimeError("프로젝트 루트에서 'data' 폴더를 찾을 수 없습니다.")
        current = current.parent

    # 2) 루트 기준으로 PDF 폴더 경로 조합
    # total data 폴더 경로
    # pdf_dir = current / "data" / "R24-11" / "AP"
    # minimum data 폴더 경로
    pdf_dir = current / "data" / "minimum"

    # 3) glob으로 PDF 목록 수집
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise ValueError(f"No PDF files found in {pdf_dir}")
    # 4) 문자열 리스트로 반환
    return [str(p) for p in pdfs]


# 기본 경로("/")에 대한 리다이렉션 처리
@app.get("/", response_class=HTMLResponse)
async def root_with_welcome():
    welcome = RagChain(file_paths=get_rag_pdfs()).get_welcome_message()
    # 간단한 HTML로 환영 메시지와 playground로 이동 버튼 제공
    return f"""
    <html>
        <head>
            <meta charset='utf-8'>
            <title>선율RAG 환영</title>
        </head>
        <body style='font-family:sans-serif;text-align:center;margin-top:10em;'>
            <h2>{welcome}</h2>
            <a href='/chat/playground' style='display:inline-block;margin-top:2em;padding:1em 2em;background:#0078d4;color:white;text-decoration:none;border-radius:8px;font-size:1.2em;'>채팅 시작하기</a>
        </body>
    </html>
    """


# translate 체인 추가
add_routes(app, Translator().create(), path="/translate")

# llm 체인 추가
add_routes(app, LLM().create(), path="/llm")

# topic 체인 추가
add_routes(app, TopicChain().create(), path="/topic")

# RAG 체인 추가
add_routes(
    app,
    RagChain(file_paths=get_rag_pdfs()).create(),
    path="/rag",
)


########### 대화형 인터페이스 ###########


class InputChat(BaseModel):
    """채팅 입력을 위한 기본 모델 정의"""

    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )


# 대화형 채팅 엔드포인트 설정
# LangSmith를 사용하는 경우, 경로에 enable_feedback_endpoint=True 을 설정하여 각 메시지 뒤에 엄지척 버튼을 활성화하고
# enable_public_trace_link_endpoint=True 을 설정하여 실행에 대한 공개 추적을 생성하는 버튼을 추가할 수도 있습니다.
# LangSmith 관련 환경 변수를 설정해야 합니다(.env)
# add_routes(
#     app,
#     # ChatChain().create().with_types(input_type=InputChat),
#     RagChatChain(file_path="data/AUTOSAR_AP_EXP_PlatformDesign.pdf").create().with_types(input_type=InputChat),
#     path="/chat",
#     enable_feedback_endpoint=True,
#     enable_public_trace_link_endpoint=True,
#     playground_type="chat",
# )
add_routes(
    app,
    RagChatChain(file_paths=get_rag_pdfs()).create().with_types(input_type=InputChat),
    path="/chat",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="chat",
)

# 최초 접속 환영 메시지 엔드포인트 추가
@app.get("/chat/welcome")
async def chat_welcome():
    welcome = RagChain(file_paths=get_rag_pdfs()).get_welcome_message()
    return {"message": welcome}


# 서버 실행 설정
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
