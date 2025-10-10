# rag/connector/embedding/ALIYUN.py
import os
from typing import List, Optional, Any
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.embeddings import Embeddings


load_dotenv()
DASHSCOPE_API_KEY_FROM_ENV = os.getenv("DASHSCOPE_API_KEY")
ALIYUN_BASE_URL_FROM_ENV = os.getenv("ALIYUN_BASE_URL")
EMBEDDING_MODEL_NAME_FROM_ENV = os.getenv("EMBEDDING_MODEL_NAME")

class QwenEmbeddings(Embeddings):
    """
    阿里云通义千问Embedding模型
    """
    api_key: Optional[str] = DASHSCOPE_API_KEY_FROM_ENV
    model_name: str = EMBEDDING_MODEL_NAME_FROM_ENV 
    base_url: str = ALIYUN_BASE_URL_FROM_ENV
    embedding_dim: int = 512  

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not (self.api_key and self.api_key.strip()):
            raise ValueError(
                "请提供DASHSCOPE_API_KEY！可选方式：\n"
                "1. 实例化QwenEmbeddings时传入 api_key='你的密钥'\n"
                "2. 在.env文件中配置 DASHSCOPE_API_KEY='你的密钥'"
            )

    @property
    def _client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def embed_query(self, query: str) -> List[float]:
        """
        QUERY转向量
        :param query: 用户输入的查询字符串
        :return: 长度为embedding_dim的向量列表
        """
        try:
            clean_query = query.strip()
            if not clean_query:
                return [0.0] * self.embedding_dim  

            response = self._client.embeddings.create(
                model=self.model_name,
                input=clean_query
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"通义Embedding查询向量生成失败: {str(e)}") from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        文本转向量
        :param texts: 文档拆分后的文本列表
        :return: 向量列表
        """
        try:
            embeddings = []
            for text in texts:
                clean_text = text.strip()
                if not clean_text:
                    embeddings.append([0.0] * self.embedding_dim)
                    continue

                response = self._client.embeddings.create(
                    model=self.model_name,
                    input=clean_text
                )
                embeddings.append(response.data[0].embedding)
            return embeddings
        except Exception as e:
            raise RuntimeError(f"通义Embedding文档向量批量生成失败: {str(e)}") from e


