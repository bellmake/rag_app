from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import load_prompt
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage
from pathlib import Path
import hashlib

from base import BaseChain

# 공통 문서 포맷터
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        try:
            page_num = doc.metadata.get("page_number", "unknown")
            source = doc.metadata.get("source_file", doc.metadata.get("source", ""))
            formatted_doc = (
                f"<document>"
                f"<content>{doc.page_content}</content>"
                f"<page>{page_num}</page>"
                f"<source>{source}</source>"
                f"</document>"
            )
            formatted_docs.append(formatted_doc)
        except Exception as e:
            formatted_docs.append(
                f"<document><content>Error formatting document: {str(e)}</content>"
                f"<page>error</page><source>error</source></document>"
            )
    return "\n\n".join(formatted_docs)

# === 환영 메시지 분리 ===
def get_welcome_message() -> str:
    """웹 루트 및 챗 환영용 메시지를 반환"""
    return "안녕하세요, 선율RAG 입니다. 어댑티브 오토사에 대해 무엇이든 물어보세요!"

# class RagChain(BaseChain):
#     """
#     RAG 기반 체인 (문서 검색 및 QA)
#     """
#     def __init__(
#         self,
#         model: str = "exaone3.5:7.8b",
#         temperature: float = 0.3,
#         system_prompt: Optional[str] = None,
#         **kwargs,
#     ):
#         super().__init__(model, temperature, **kwargs)
#         self.welcome_message = "안녕하세요, 선율RAG 입니다. 어댑티브 오토사에 대해 무엇이든 물어보세요!"
#         self.system_prompt = (
#             system_prompt or "You are a helpful AI Assistant. Your name is '선율'. You must answer in Korean."
#         )
#         if "file_paths" in kwargs:
#             self.file_paths = kwargs.pop("file_paths")
#         elif "file_path" in kwargs:
#             self.file_paths = [kwargs.pop("file_path")]
#         else:
#             raise ValueError("file_path(s) is required")

#     def get_welcome_message(self):
#         """최초 접속시 보여줄 환영 메시지 반환"""
#         return self.welcome_message

#     def setup(self):
#         if not self.file_paths:
#             raise ValueError("file_path(s) is required")
#         print("RagChain setup")

#         # PDF 로딩 및 분할
#         raw_docs = []
#         for file_path in self.file_paths:
#             loader = PDFPlumberLoader(file_path)
#             docs_from_file = loader.load()
#             for i, doc in enumerate(docs_from_file):
#                 doc.metadata["page_number"] = doc.metadata.get("page_number", i + 1)
#                 doc.metadata["source_file"] = file_path
#             raw_docs.extend(docs_from_file)
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         docs = text_splitter.split_documents(raw_docs)

#         # 파일 목록 해시 기반 캐시 디렉토리 설정
#         file_key = "|".join(sorted(self.file_paths))
#         cache_root = Path(__file__).parent.parent / "embedding_cache"
#         cache_id = hashlib.md5(file_key.encode()).hexdigest()
#         faiss_dir = cache_root / cache_id
#         faiss_dir.mkdir(parents=True, exist_ok=True)

#         # 임베딩 생성 및 FAISS 인덱스 로드/생성
#         EMBEDDING_MODEL = "bge-m3"
#         embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
#         if any(faiss_dir.iterdir()):  # 디렉토리에 파일이 있으면 로드
#             vectorstore = FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)
#         else:
#             vectorstore = FAISS.from_documents(docs, embedding=embeddings)
#             vectorstore.save_local(str(faiss_dir))

#         retriever = vectorstore.as_retriever()

#         # 프롬프트 및 LLM 설정
#         prompt = load_prompt("prompts/rag-llama.yaml", encoding="utf-8")
#         llm = ChatOllama(
#             model=self.model,
#             temperature=self.temperature,
#         )

#         # 메시지 결합 함수
#         def combine_messages(input_dict):
#             messages = input_dict["messages"]
#             last_user_message = next(
#                 (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None
#             )
#             if last_user_message:
#                 context_docs = retriever.get_relevant_documents(last_user_message.content)
#                 return {
#                     "question": last_user_message.content,
#                     "context": format_docs(context_docs),
#                 }
#             else:
#                 return {"question": "", "context": ""}

#         # 체인 구성
#         chain = (
#             RunnablePassthrough()
#             | combine_messages
#             | prompt
#             | llm
#             | StrOutputParser()
#         )
#         return chain
