from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import load_prompt
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.messages import HumanMessage

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

class RagChain(BaseChain):
    """
    RAG 기반 체인 (문서 검색 및 QA)
    """
    def __init__(
        self,
        model: str = "exaone3.5:32b",
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.welcome_message = "안녕하세요, 선율RAG 입니다. 어댑티브 오토사에 대해 무엇이든 물어보세요!"
        self.system_prompt = (
            system_prompt or "You are a helpful AI Assistant. Your name is '선율'. You must answer in Korean."
        )
        if "file_paths" in kwargs:
            self.file_paths = kwargs.pop("file_paths")
        elif "file_path" in kwargs:
            self.file_paths = [kwargs.pop("file_path")]
        else:
            raise ValueError("file_path(s) is required")

    def get_welcome_message(self):
        """최초 접속시 보여줄 환영 메시지 반환"""
        return self.welcome_message

    def setup(self):
        if not self.file_paths:
            raise ValueError("file_path(s) is required")
        
        print("RagChain setup")

        # 1) PDF 로딩 (여러 파일을 모두 읽어서 raw_docs에 합치기)
        raw_docs = []
        for file_path in self.file_paths:
            loader = PDFPlumberLoader(file_path)
            docs_from_file = loader.load()
            # 각 문서에 원본 파일 정보 추가 (출처 정보)
            for i, doc in enumerate(docs_from_file):
                doc.metadata["page_number"] = doc.metadata.get("page_number", i + 1)
                doc.metadata["source_file"] = file_path
            raw_docs.extend(docs_from_file)

        # 2) (선택 사항) 전체 문서에 대해 페이지 번호를 재설정할 수도 있지만,
        #    원본 페이지 번호를 유지하고 싶다면 생략하거나 파일별로 따로 관리할 수 있습니다.
        # 아래 코드는 전체 문서 리스트에서 번호가 누락된 경우에만 부여합니다.
        for i, doc in enumerate(raw_docs):
            if "page_number" not in doc.metadata:
                doc.metadata["page_number"] = i + 1

        # 3) Text Splitter로 문서 chunk 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        docs = text_splitter.split_documents(raw_docs)
        # 각 chunk는 원본 문서의 metadata(예: page_number, source_file)를 그대로 유지함

        # 4) 캐싱을 지원하는 임베딩 설정
        EMBEDDING_MODEL = "bge-m3"
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # 5) 벡터 DB 생성
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        
        # 6) 문서 검색기 설정
        retriever = vectorstore.as_retriever()
        
        # 7) 프롬프트 로드
        prompt = load_prompt("prompts/rag-llama.yaml", encoding="utf-8")
        
        # 8) Ollama 모델 생성
        llm = ChatOllama(
            model="exaone3.5:32b",
            temperature=0,
        )
        
        # 9) 사용자 질문으로부터 관련 문서 검색 후 QA 요청 데이터 생성
        def combine_messages(input_dict):
            messages = input_dict["messages"]
            last_user_message = next(
                (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None
            )
            if last_user_message:
                context_docs = retriever.get_relevant_documents(last_user_message.content)
                return {
                    "question": last_user_message.content,
                    "context": format_docs(context_docs),
                }
            else:
                return {
                    "question": "",
                    "context": "",
                }

        # 10) 체인 생성
        chain = (
            RunnablePassthrough()
            | combine_messages
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain

