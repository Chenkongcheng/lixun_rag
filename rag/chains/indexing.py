import os
import sys
import logging
from datetime import datetime

current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.insert(0, project_root)

from typing import List, Optional, Any
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rag.chains.base import BaseIndexingChain
from rag.connector.vectorstore.chroma_store import ChromaVectorStore
from rag.module.loader.loader_manager import loader_manager

# 配置日志
logger = logging.getLogger(__name__)


class MultiTypeIndexingChain(BaseIndexingChain):
    """
    多类型文档的索引链（支持PDF、Word、Markdown、TXT等）
    功能：自动识别文件类型 → 加载内容 → 拆分文本 → 存入向量库
    """
    def __init__(
        self,
        vector_store: ChromaVectorStore,
        splitter: Optional[RecursiveCharacterTextSplitter] = None,** kwargs: Any
    ):
        super().__init__(**kwargs)
        self.vector_store = vector_store
        self.splitter = splitter or RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )

    def load(self, file_paths: List[str]) -> List[Document]:
        """
        加载多种类型的文档
        :param file_paths: 文件路径
        :return: 统一的Document列表
        """
        if not file_paths or not all(path.strip() for path in file_paths):
            raise ValueError("文件路径无效！")
        
        try:
            # 使用加载器管理器加载所有文件
            docs = loader_manager.load_multiple(file_paths)
            
            # 为文档添加额外元数据
            for doc in docs:
                doc.metadata["load_time"] = self._get_current_time()
                # 如果没有来源信息，从文件路径提取
                if "source" not in doc.metadata and "file_path" in doc.metadata:
                    doc.metadata["source"] = doc.metadata["file_path"]
            
            logger.info(f"成功加载{len(docs)}个文档（包含所有类型）")
            return docs
        except Exception as e:
            raise RuntimeError(f"文档加载失败（路径：{file_paths[:1]}...）: {str(e)}") from e

    def split(self, docs: List[Document], splitter: Optional[RecursiveCharacterTextSplitter] = None) -> List[Document]:
        """拆分文档"""
        if not docs:
            raise ValueError("文档列表不能为空！")
        
        used_splitter = splitter or self.splitter
        try:
            split_docs = used_splitter.split_documents(docs)
            for i, doc in enumerate(split_docs):
                doc.metadata["chunk_id"] = i
                doc.metadata["total_chunks"] = len(split_docs)
            
            logger.info(f"文档拆分完成：{len(docs)}个原始文档 → {len(split_docs)}个chunk")
            return split_docs
        except Exception as e:
            raise RuntimeError(f"文档拆分失败: {str(e)}") from e

    def store(self, chunks: List[Document]) -> List[str]:
        """存储文档片段"""
        if not chunks:
            raise ValueError("待存储的文档片段列表不能为空！")
        
        try:
            doc_ids = self.vector_store.add_documents(documents=chunks)
            logger.info(f"文档存储完成：{len(chunks)}个chunk已存入向量库，集合名称：{self.vector_store.collection_name}")
            return doc_ids
        except Exception as e:
            raise RuntimeError(f"文档存储失败: {str(e)}") from e

    def run(self, file_paths: List[str]) -> List[str]:
        """执行完整索引流程"""

        docs = self.load(file_paths)
        chunks = self.split(docs)
        doc_ids = self.store(chunks)
        
        return doc_ids

    @staticmethod
    def _get_current_time() -> str:
        """获取当前时间，用于metadata记录"""
        return datetime.now().isoformat()



    