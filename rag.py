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
        model: str = "llama2:13b",
        # model: str = "llama3.1:70b",       
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt or "You are a helpful AI Assistant. Your name is '선율'. You must answer in Korean."
        )
        if "file_path" in kwargs:
            self.file_path = kwargs["file_path"]

    def setup(self):
        if not self.file_path:
            raise ValueError("file_path is required")
        
        # Splitter 설정
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        
        # 문서 로드
        loader = PDFPlumberLoader(self.file_path)
        # 전체 문서를 먼저 로드
        raw_docs = loader.load()
        
        # 각 페이지의 실제 페이지 번호를 저장
        page_numbers = {}
        for i, doc in enumerate(raw_docs):
            # PDFPlumber의 page_number는 1부터 시작하는 실제 PDF 페이지 번호
            actual_page = doc.metadata.get('page_number', i + 1)
            page_numbers[i] = actual_page
        
        # 문서 분할
        docs = text_splitter.split_documents(raw_docs)
        
        # 문서에 실제 페이지 정보 추가
        for doc in docs:
            # 원본 청크의 인덱스를 사용하여 실제 페이지 번호 매핑
            original_page_idx = doc.metadata.get('page', 0)
            doc.metadata['page'] = page_numbers.get(original_page_idx, original_page_idx + 1)

        # 캐싱을 지원하는 임베딩 설정
        EMBEDDING_MODEL = "bge-m3"
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # 벡터 DB 저장
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        
        # 문서 검색기 설정
        retriever = vectorstore.as_retriever()
        
        # 프롬프트 로드 (프롬프트 파일을 rag-llama2-13b.yaml로 변경)
        prompt = load_prompt("prompts/rag-llama.yaml", encoding="utf-8")
        
        # Ollama 모델 지정 (모델명을 llama2-13b로 변경)
        llm = ChatOllama(
            model="llama2:13b",
            # model="llama3.1:70b",
            temperature=0,
        )
        
        # 체인 생성
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
