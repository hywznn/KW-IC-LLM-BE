from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class Chatbot:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        self.qa_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def initialize_vectorstore(self, texts):
        """텍스트 데이터로 벡터 스토어를 초기화합니다."""
        self.vectorstore = FAISS.from_texts(texts, self.embeddings)
        self._initialize_qa_chain()

    def save_vectorstore(self, path: str = "vectorstore"):
        """벡터 스토어를 파일로 저장합니다."""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            return True
        return False

    def load_vectorstore(self, path: str = "vectorstore"):
        """저장된 벡터 스토어를 불러옵니다."""
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(path, self.embeddings)
            self._initialize_qa_chain()
            return True
        return False

    def _initialize_qa_chain(self):
        """QA 체인을 초기화합니다."""
        if self.vectorstore:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0.7),
                retriever=self.vectorstore.as_retriever(),
                memory=self.memory,
            )

    def get_response(self, query: str) -> str:
        """사용자 쿼리에 대한 응답을 생성합니다."""
        if not self.qa_chain:
            return "벡터 스토어가 초기화되지 않았습니다."
        
        response = self.qa_chain({"question": query})
        return response["answer"] 