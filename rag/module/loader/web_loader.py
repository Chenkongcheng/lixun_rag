from typing import List, Iterator, Union
import bs4
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from .base_loader import BaseLoader


class CustomizedWebLoader(WebBaseLoader, BaseLoader):
    """网页内容加载器"""
    
    def __init__(self, web_path: Union[str, List[str]]):
        super().__init__(
            web_paths=web_path,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header", "article", "main")
                )
            )
        )
        # 对于BaseLoader需要文件路径，这里用网页URL代替
        self.file_path = web_path if isinstance(web_path, str) else web_path[0]

    def load(self) -> List[Document]:
        """加载网页内容"""
        docs = super().load()
        # 为每个文档添加来源信息
        for doc, url in zip(docs, self.web_paths):
            doc.metadata["source"] = url
            doc.metadata["type"] = "webpage"
        return docs

    def lazy_load(self) -> Iterator[Document]:
        """懒加载网页内容"""
        for doc in super().lazy_load():
            yield doc

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """支持的URL前缀"""
        return ["http://", "https://"]

    