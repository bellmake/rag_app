from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from typing import Optional, List
from base import BaseChain
import time
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager

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

class TokenSpeedCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.start_time = time.time()
        self.token_count = 0
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.token_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            tokens_per_sec = self.token_count / elapsed
            print(f"토큰 속도: {tokens_per_sec:.2f} tokens/s")

callback_handler = TokenSpeedCallbackHandler()
callback_manager = CallbackManager([callback_handler])

class TopicChain(BaseChain):
    """
    주어진 주제에 대해 설명하는 체인
    """
    def __init__(self, model: str = "exaone", temperature: float = 0, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = system_prompt or "You are a helpful assistant. Your mission is to explain given topic in a concise manner. Answer in Korean."
    def setup(self):
        llm = ChatOllama(model=self.model, temperature=self.temperature)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "Here is the topic: {topic}"),
        ])
        return prompt | llm | StrOutputParser()

class ChatChain(BaseChain):
    """
    대화형 체인
    """
    def __init__(self, model: str = "exaone", temperature: float = 0.3, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = system_prompt or "You are a helpful AI Assistant. Your name is '선율'. You must answer in Korean."
    def setup(self):
        llm = ChatOllama(model=self.model, temperature=self.temperature)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        return prompt | llm | StrOutputParser()

class LLM(BaseChain):
    """
    프롬프트 없이 LLM만 반환하는 체인
    """
    def setup(self):
        return ChatOllama(model=self.model, temperature=self.temperature)

class Translator(BaseChain):
    """
    번역 체인 (한국어 번역)
    """
    def __init__(self, model: str = "exaone", temperature: float = 0, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = system_prompt or "You are a helpful assistant. Your mission is to translate given sentences into Korean."
    def setup(self):
        llm = ChatOllama(model=self.model, temperature=self.temperature)
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", "Here is the sentence: {input}"),
        ])
        return prompt | llm | StrOutputParser()

class RagChatChain(BaseChain):
    """
    RAG 기반 대화형 체인
    """
    def __init__(self, model: str = "exaone3.5:32b", temperature: float = 0.3, system_prompt: Optional[str] = None, **kwargs):
        super().__init__(model, temperature, **kwargs)
        self.system_prompt = system_prompt or "You are a helpful AI Assistant. You must Answer in Korean. Your name is '선율'."
        if "file_paths" in kwargs:
            self.file_paths = kwargs.pop("file_paths")
        elif "file_path" in kwargs:
            self.file_paths = [kwargs.pop("file_path")]
        else:
            raise ValueError("file_path(s) is required")
        self.vectorstore = None
    def setup(self):
        if not self.file_paths:
            raise ValueError("file_paths is required")
        print("RagChatChain setup")
        raw_docs = []
        for file_path in self.file_paths:
            loader = PDFPlumberLoader(file_path)
            docs_from_file = loader.load()
            for i, doc in enumerate(docs_from_file):
                doc.metadata["page_number"] = doc.metadata.get("page_number", i + 1)
                doc.metadata["source_file"] = file_path
            raw_docs.extend(docs_from_file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(raw_docs)
        embeddings = OllamaEmbeddings(model="bge-m3")
        self.vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        prompt = load_prompt("prompts/rag-llama.yaml", encoding="utf-8")
        llm = ChatOllama(
            model="exaone3.5:32b",
            temperature=0,
            callback_manager=callback_manager,
            streaming=True
        )
        def combine_messages(input_dict):
            messages = input_dict["messages"]
            last_user_message = next((msg for msg in reversed(messages) if isinstance(msg, HumanMessage)), None)
            if last_user_message:
                context_docs = self.vectorstore.as_retriever().get_relevant_documents(last_user_message.content)
                return {
                    "question": last_user_message.content,
                    "context": format_docs(context_docs),
                }
            else:
                return {"question": "", "context": ""}
        chain = RunnablePassthrough() | combine_messages | prompt | llm | StrOutputParser()
        return chain