import os
from typing import List, Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from .base_loader import BaseLoader


class CustomizedTxtLoader(TextLoader, BaseLoader):
    """TXT文档加载器"""
    
    def __init__(self, file_path: str, encoding: str = "utf-8"):
        TextLoader.__init__(self, file_path=file_path, encoding=encoding)
        BaseLoader.__init__(self, file_path=file_path)

    def load(self) -> List[Document]:
        """加载文档"""
        return super().load()

    def lazy_load(self) -> Iterator[Document]:
        """懒加载文档"""
        for doc in super().lazy_load():
            yield doc

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """支持的文件扩展名"""
        return [".txt"]

    