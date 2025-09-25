from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage
from rag.chains.base import BaseGenerationChain
from rag.connector.llm.ALIYUN import QwenLLM


class BasicGenerationChain(BaseGenerationChain):
    """
    生成答案，支持多轮对话
    """
    def __init__(
        self,
        llm: Optional[QwenLLM] = None,** kwargs: Any
    ):
        super().__init__(**kwargs)
        self.llm = llm or QwenLLM()
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "context", "question"],
            template="""请基于以下上下文和对话历史回答用户的最新问题。
如果上下文没有相关信息，直接说"没有找到相关信息"。

对话历史:
{chat_history}

上下文:
{context}

最新问题: {question}

回答:"""
        )

    def augment(
        self, 
        query: str, 
        docs: List[Document], 
        chat_history: List[BaseMessage] = None
    ) -> str:
        """构建包含对话历史的增强提示词"""
        # 格式化文档上下文
        context = "\n\n".join([
            f"[来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}"
            for doc in docs
        ])
        
        # 格式化对话历史
        history_str = ""
        if chat_history:
            for i, msg in enumerate(chat_history):
                role = "用户" if i % 2 == 0 else "助手"
                history_str += f"{role}: {msg.content}\n"
        
        # 生成最终提示
        augmented_prompt = self.prompt.format(
            chat_history=history_str,
            context=context,
            question=query
        )
        return augmented_prompt

    def generate(self, prompt: str) -> str:
        """调用LLM生成回答"""
        if not prompt.strip():
            raise ValueError("提示词不能为空！")
        
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            raise RuntimeError(f"LLM生成回答失败: {str(e)}") from e
