# rag/chains/retrieval.py
from typing import List, Optional, Any
from langchain_core.documents import Document
from rag.chains.base import BaseRetrievalChain
from rag.connector.vectorstore.ChromaStore import ChromaVectorStore


class BasicRetrievalChain(BaseRetrievalChain):
    """
    基础召回链（实现BaseRetrievalChain）
    功能：预处理查询 → 向量检索相关文档 → 后处理结果
    """
    def __init__(
        self,
        vector_store: ChromaVectorStore,  
        top_k: int = 3,  
        **kwargs: Any
    ):
        super().__init__(** kwargs)
        self.vector_store = vector_store
        self.top_k = top_k  

    def pre_retrieval(self, query: str) -> str:
        """
        检索前预处理query
        功能：清洗查询、标准化格式，为检索做准备
        :param query: 原始用户查询
        :return: 预处理后的查询
        """
        if not query:
            raise ValueError("查询不能为空！")
        
        # 基础清洗：去除首尾空格、多余空行
        cleaned_query = query.strip()
        
        # 可扩展：添加查询扩展（如同义词替换）、多语言翻译等逻辑
        # 示例：若查询过短（<5字符），添加默认前缀提升检索准确性
        if len(cleaned_query) < 5:
            cleaned_query = f"详细解释一下：{cleaned_query}"
        
        return cleaned_query

    def retrieval(self, query: str) -> List[Document]:
        """
        执行检索
        功能：调用向量库，根据查询获取相关文档
        :param query: 预处理后的查询
        :return: 相关文档列表（按相关性排序）
        """
        try:
            relevant_docs = self.vector_store.similarity_search(
                query=query,
                k=self.top_k
            )
            return relevant_docs
        except Exception as e:
            raise RuntimeError(f"检索相关文档失败（查询：{query[:20]}...）: {str(e)}") from e

    def post_retrieval(self, query: str, docs: List[Document]) -> List[Document]:
        """
        检索后处理结果
        功能：过滤低质量文档、去重、补充元数据等
        :param query: 预处理后的查询
        :param docs: 检索到的原始文档列表
        :return: 处理后的高质量文档列表
        """
        if not docs:
            return []
        
        # 1. 过滤过短文档（避免无意义片段，如仅几个字符）
        filtered_docs = [doc for doc in docs if len(doc.page_content.strip()) > 5]
        
        # 2. 简单去重
        seen_contents = set()
        unique_docs = []
        for doc in filtered_docs:
            content_hash = hash(doc.page_content.strip())
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_docs.append(doc)
        
        # 3. 为文档添加检索分数
        for i, doc in enumerate(unique_docs):
            doc.metadata["retrieval_rank"] = i + 1  
        
        return unique_docs


