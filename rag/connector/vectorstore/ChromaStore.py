from typing import List, Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rag.connector.embedding.ALIYUN import QwenEmbeddings


class ChromaVectorStore:
    """
    Chroma向量数据库的封装类
    功能：初始化数据库、添加文档向量、检索相关文档、删除集合
    """
    def __init__(
        self,
        collection_name: str = "default_rag_collection",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[QwenEmbeddings] = None,
        **chroma_kwargs: Any
    ):
        self.embedding_model = embedding_model or QwenEmbeddings()
        self.persist_directory = persist_directory  # 保存持久化目录参数
        self.chroma_client = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=persist_directory,
            create_collection_if_not_exists=True,** chroma_kwargs
        )
        self.collection_name = collection_name

    def add_documents(self, documents: List[Document]) -> List[str]:
        """向向量库添加文档"""
        if not documents:
            raise ValueError("添加的文档列表不能为空！")
        
        try:
            doc_ids = self.chroma_client.add_documents(documents=documents)
            if self.persist_directory:
                self.chroma_client.persist()  
            return doc_ids
        except Exception as e:
            raise RuntimeError(f"Chroma添加文档失败（集合：{self.collection_name}）: {str(e)}") from e

    def similarity_search(
        self,
        query: str,
        k: int = 3
    ) -> List[Document]:
        """语义检索相关文档"""
        if not query.strip():
            raise ValueError("检索查询不能为空！")
        
        try:
            relevant_docs = self.chroma_client.similarity_search(
                query=query.strip(),
                k=k
            )
            return relevant_docs
        except Exception as e:
            raise RuntimeError(f"Chroma检索文档失败（查询：{query[:20]}...）: {str(e)}") from e

    def delete_collection(self) -> None:
        """删除当前集合"""
        try:
            self.chroma_client.delete_collection()
        except Exception as e:
            raise RuntimeError(f"Chroma删除集合失败（集合：{self.collection_name}）: {str(e)}") from e

    @property
    def document_count(self) -> int:
        """获取当前集合的文档数量"""
        return self.chroma_client._collection.count()


