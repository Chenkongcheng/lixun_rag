"""RAG核心链模块：包含索引、检索、生成的完整流程"""
from .base import BaseIndexingChain, BaseRetrievalChain, BaseGenerationChain
from .indexing import MultiTypeIndexingChain  # 改为新的类名
from .retrieval import BasicRetrievalChain
from .generate import BasicGenerationChain

__all__ = [
    "BaseIndexingChain", "BaseRetrievalChain", "BaseGenerationChain",
    "MultiTypeIndexingChain",  # 同步更新导出列表
    "BasicRetrievalChain", "BasicGenerationChain"
]
