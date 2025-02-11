from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import load_prompt


from typing import Optional, List
from base import BaseChain

# def format_docs(docs):
#     return "\n\n".join(
#         f"<document><content>{doc.page_content}</content>"
#         f"<page>{doc.metadata.get('page_number', 'unknown')}</page>"
#         f"<source>{doc.metadata.get('source', '')}</source></document>"
#         for doc in docs
#     )

class TopicChain(BaseChain):
    """
    주어진 주제에 대해 설명하는 체인 클래스입니다.

    Attributes:
        model (str): 사용할 LLM 모델명
        temperature (float): 모델의 temperature 값
        system_prompt (str): 시스템 프롬프트
    """

    def __init__(
        self,
        model: str = "exaone",
        temperature: float = 0,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant. Your mission is to explain given topic in a concise manner. Answer in Korean."
        )

    def setup(self):
        """TopicChain을 설정하고 반환합니다."""
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "Here is the topic: {topic}"),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        return chain


class ChatChain(BaseChain):
    """
    대화형 체인 클래스입니다.

    Attributes:
        model (str): 사용할 LLM 모델명
        temperature (float): 모델의 temperature 값
        system_prompt (str): 시스템 프롬프트
    """

    def __init__(
        self,
        model: str = "exaone",
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt
            or "You are a helpful AI Assistant. Your name is '선율'. You must answer in Korean."
        )

    def setup(self):
        """ChatChain을 설정하고 반환합니다."""
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        return chain


class LLM(BaseChain):
    """
    기본 LLM 체인 클래스입니다.
    다른 체인들과 달리 프롬프트 없이 직접 LLM을 반환합니다.
    """

    def setup(self):
        """LLM 인스턴스를 설정하고 반환합니다."""
        llm = ChatOllama(model=self.model, temperature=self.temperature)
        return llm


class Translator(BaseChain):
    """
    번역 체인 클래스입니다.
    주어진 문장을 한국어로 번역합니다.

    Attributes:
        model (str): 사용할 LLM 모델명
        temperature (float): 모델의 temperature 값
        system_prompt (str): 시스템 프롬프트
    """

    def __init__(
        self,
        model: str = "exaone",
        temperature: float = 0,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt
            or "You are a helpful assistant. Your mission is to translate given sentences into Korean."
        )

    def setup(self):
        """Translator 체인을 설정하고 반환합니다."""
        llm = ChatOllama(model=self.model, temperature=self.temperature)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", "Here is the sentence: {input}"),
            ]
        )

        chain = prompt | llm | StrOutputParser()
        return chain


class RagChatChain(BaseChain):
    def __init__(
        self,
        # model: str = "llama2:13b",
        model: str = "llama3.1:70b",
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = (
            system_prompt
            or "You are a helpful AI Assistant. You must Answer in Korean. Your name is '선율'. You must answer in Korean."
        )
        # file_path가 kwargs로 들어왔는지 확인
        if "file_path" in kwargs:
            self.file_path = kwargs["file_path"]
        else:
            raise ValueError("file_path is required")

        # Vectorstore 미리 None 초기화
        self.vectorstore = None
    
    def setup(self):
        if not self.file_path:
            raise ValueError("file_path is required")
        
        print("RagChatChain setup")

        # 1) PDF 로딩 (LangChain의 PDFPlumberLoader 사용)
        loader = PDFPlumberLoader(self.file_path)
        raw_docs = loader.load()

        # 2) 페이지 번호 메타데이터 설정
        #    (LangChain의 PDFPlumberLoader가 page_number를 넣어줄 수도 있지만,
        #     혹시 없는 경우를 대비해 manual하게 설정)
        for i, doc in enumerate(raw_docs):
            doc.metadata["page_number"] = doc.metadata.get("page_number", i + 1)

        # 3) Text Splitter로 문서 chunk 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        docs = text_splitter.split_documents(raw_docs)
        # 이때 분할된 chunk들도 원본 doc.metadata를 자동으로 이어받습니다.
        # 즉, 각 chunk의 doc.metadata["page_number"]가 유지됨

        # 4) Embeddings & VectorStore 생성
        EMBEDDING_MODEL = "bge-m3"
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.vectorstore = FAISS.from_documents(docs, embedding=embeddings)

        # 5) LLM & Prompt
        # prompt = load_prompt("prompts/rag-llama2-13b.yaml", encoding="utf-8")
        # llm = ChatOllama(model="llama2:13b", temperature=0)
        prompt = load_prompt("prompts/rag-llama.yaml", encoding="utf-8")
        llm = ChatOllama(model="llama3.1:70b", temperature=0)

        # 6) 포맷팅 함수 (검색된 문서 chunk를 모델에 넘길 때)
        def format_docs(docs):
            formatted_docs = []
            for doc in docs:
                try:
                    page_num = doc.metadata.get("page_number", "unknown")
                    source = doc.metadata.get("source", "")
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

        # 7) 사용자 입력(마지막 유저 메시지)으로부터 관련 문서 검색 -> QA
        def combine_messages(input_dict):
            messages = input_dict["messages"]
            last_user_message = next(
                (msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None
            )
            if last_user_message:
                context_docs = self.vectorstore.as_retriever().get_relevant_documents(
                    last_user_message.content
                )
                return {
                    "question": last_user_message.content,
                    "context": format_docs(context_docs),
                }
            else:
                return {
                    "question": "",
                    "context": "",
                }

        # 체인 연결
        chain = (
            RunnablePassthrough()
            | combine_messages
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain