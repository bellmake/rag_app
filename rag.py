from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import load_prompt
from langchain_ollama import OllamaEmbeddings, ChatOllama

from base import BaseChain


# 문서 포맷팅
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        try:
            # 페이지 번호가 없는 경우 'unknown'으로 처리
            page_num = doc.metadata.get('page', 'unknown')
            # 소스 정보가 없는 경우 빈 문자열로 처리
            source = doc.metadata.get('source', '')
            
            formatted_doc = (
                f"<document>"
                f"<content>{doc.page_content}</content>"
                f"<page>{page_num}</page>"
                f"<source>{source}</source>"
                f"</document>"
            )
            formatted_docs.append(formatted_doc)
        except Exception as e:
            # 에러가 발생한 문서는 건너뛰되, 에러 정보를 포함
            formatted_docs.append(
                f"<document><content>Error formatting document: {str(e)}</content>"
                f"<page>error</page><source>error</source></document>"
            )
    
    return "\n\n".join(formatted_docs)

class RagChain(BaseChain):
    def __init__(
        self,
        # model: str = "llama2:13b",
        model: str = "exaone3.5:32b",       
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt or "You are a helpful AI Assistant. Your name is '선율'. You must answer in Korean."
        )
        if "file_paths" in kwargs:
            self.file_paths = kwargs.pop("file_paths")
        elif "file_path" in kwargs:
            self.file_paths = [kwargs.pop("file_path")]
        else:
            raise ValueError("file_path(s) is required")

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
                # 만약 PDFPlumberLoader가 page_number 정보를 제공하지 않는다면,
                # 각 파일 내에서 개별적으로 번호를 부여할 수 있습니다.
                if "page_number" not in doc.metadata:
                    doc.metadata["page_number"] = i + 1
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
            # model="llama2:13b",
            model="exaone3.5:32b",
            temperature=0,
        )
        
        # 9) 포맷팅 함수: 검색된 문서 chunk들을 하나의 문자열로 변환할 때,
        #    원본 문서의 페이지 번호 및 출처 정보를 포함할 수 있도록 합니다.
        def format_docs(docs):
            formatted_docs = []
            for doc in docs:
                try:
                    page_num = doc.metadata.get("page_number", "unknown")
                    source = doc.metadata.get("source_file", "")
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

        # 10) 사용자 질문으로부터 관련 문서 검색 후 QA 요청 데이터 생성
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

        # 11) 체인 생성
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain

