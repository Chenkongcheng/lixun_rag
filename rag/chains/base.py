from abc import ABC, abstractmethod
from typing import Generator
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class BaseIndexingChain(ABC):

    @abstractmethod
    def load(self, file_paths: List[str]) -> List[Document]:
        pass

    @abstractmethod
    def split(self, docs: List[Document], splitter: Any = None) -> List[Document]:
        pass

    @abstractmethod
    def store(self, chunks: List[Document]) -> List[str]:
        pass

class BaseRetrievalChain(ABC):

    @abstractmethod
    def pre_retrieval(self, query: str) -> str:
        pass

    @abstractmethod
    def retrieval(self, query: str) -> List[Document]:
        pass

    @abstractmethod
    def post_retrieval(self, query: str, docs: List[Document]) -> List[Document]:
        pass

class BaseGenerationChain(ABC):

    @abstractmethod
    def augment(self, query: str, docs: List[Document], chat_history: List[BaseMessage] = None) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass