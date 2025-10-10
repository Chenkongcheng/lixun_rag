import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List, Optional, Tuple
from langgraph.graph import StateGraph, END  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from rag.chains.indexing import MultiTypeIndexingChain
from rag.chains.retrieval import BasicRetrievalChain
from rag.chains.generate import BasicGenerationChain
from rag.connector.vectorstore.chroma_store import ChromaVectorStore
from rag.connector.llm.aliyun import QwenLLM
import logging
from langsmith import Client
from server.langsmith_setup import setup_langsmith

# 配置日志
logger = logging.getLogger(__name__)


class AgentState(BaseModel):
    file_paths: List[str] = Field(default_factory=list, description="首次索引的文件/URL")
    query: str = Field(default="", description="用户原始查询")
    reformulated_query: str = Field(default="", description="改写后的独立查询")
    chat_history: List[BaseMessage] = Field(default_factory=list, description="对话历史")
    documents: List[Any] = Field(default_factory=list, description="检索到的文档")
    answer: str = Field(default="", description="生成的回答")
    indexing_completed: bool = Field(default=False, description="索引是否已完成")


class RAGAgent:
    def __init__(self, collection_name: str = "rag_agent_collection", persist_directory: str = None):
        # 如果没有指定持久化目录，使用环境变量VECTOR_DB_PATH
        if persist_directory is None:
            import os
            persist_directory = os.getenv('VECTOR_DB_PATH', './data/chroma')
        
        self.vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        self.indexing_chain = MultiTypeIndexingChain(vector_store=self.vector_store)
        self.retrieval_chain = BasicRetrievalChain(vector_store=self.vector_store)
        self.llm = QwenLLM(temperature=0.7)
        self.generation_chain = BasicGenerationChain(llm=self.llm)
        self.contextualize_chain = self._init_contextualize_chain()
        
        # 设置LangSmith追踪
        self.tracer = setup_langsmith()
        self.langsmith_client = Client() if self.tracer else None
        
        self.graph = self._build_graph()

    def reformulate_query_node(self, state: AgentState) -> Dict[str, Any]:
        """问题改写：有历史对话则改写，无则用原始查询"""
        if not state.query:
            raise ValueError("请输入您的问题！")
        
        if state.chat_history:
            reformulated = self.contextualize_chain.invoke({
                "chat_history": state.chat_history,
                "input": state.query
            })
            logger.info(f"问题改写：原始='{state.query}' → 改写='{reformulated}'")
            return {"reformulated_query": reformulated}
        return {"reformulated_query": state.query}

    def retrieval_node(self, state: AgentState) -> Dict[str, Any]:
        """检索：从已索引向量库获取文档"""
        if not state.reformulated_query:
            raise ValueError("必须有改写后的查询才能检索")
        
        cleaned_query = self.retrieval_chain.pre_retrieval(state.reformulated_query)
        raw_docs = self.retrieval_chain.retrieval(cleaned_query)
        processed_docs = self.retrieval_chain.post_retrieval(cleaned_query, raw_docs)
        
        # 记录检索到的文档片段信息
        if processed_docs:
            logger.info(f"检索到 {len(processed_docs)} 个相关文档片段")
            for i, doc in enumerate(processed_docs, 1):
                # 获取文档内容，根据不同的文档类型处理
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                elif hasattr(doc, 'content'):
                    content = doc.content
                elif isinstance(doc, dict) and 'content' in doc:
                    content = doc['content']
                else:
                    content = str(doc)
                
                # 获取文档元数据
                metadata = {}
                if hasattr(doc, 'metadata'):
                    metadata = doc.metadata
                elif isinstance(doc, dict) and 'metadata' in doc:
                    metadata = doc.get('metadata', {})
                
                # 记录文档信息（调低日志级别避免过多输出）
                if metadata:
                    source = metadata.get('source', '未知来源')
                    page = metadata.get('page', '')
                    if page:
                        logger.debug(f"片段 {i}: 来源={source} (第{page}页), 长度={len(content)} 字符")
                    else:
                        logger.debug(f"片段 {i}: 来源={source}, 长度={len(content)} 字符")
        else:
            logger.warning("未检索到任何相关文档片段")
        
        return {"documents": processed_docs}

    def generation_node(self, state: AgentState) -> Dict[str, Any]:
        """生成回答：更新对话历史"""
        if not state.reformulated_query:
            raise ValueError("生成回答需要查询")
        
        # 如果没有检索到文档，使用空文档列表继续
        if not state.documents:
            logger.warning("未检索到文档，使用空文档生成回答")
        
        full_prompt = self.generation_chain.augment(
            query=state.reformulated_query,
            docs=state.documents,
            chat_history=state.chat_history
        )
        answer = self.generation_chain.generate(full_prompt)
        
        # 追加当前轮对话到历史
        new_history = state.chat_history + [
            HumanMessage(content=state.query),
            AIMessage(content=answer)
        ]
        return {"answer": answer, "chat_history": new_history}

    def _init_contextualize_chain(self):
        """初始化问题改写链"""
        system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        return prompt | self.llm | StrOutputParser()

    def _build_graph(self):
        """构建 LangGraph 工作流"""
        workflow = StateGraph(AgentState)

        # 注册节点
        workflow.add_node("reformulate_query", self.reformulate_query_node)
        workflow.add_node("retrieval", self.retrieval_node)
        workflow.add_node("generation", self.generation_node)

        # 设置入口点
        workflow.set_entry_point("reformulate_query")

        # 添加边
        workflow.add_edge("reformulate_query", "retrieval")
        workflow.add_edge("retrieval", "generation")
        workflow.add_edge("generation", END)

        return workflow.compile()

    def run(
        self, 
        query: str, 
        file_paths: List[str] = None, 
        chat_history: List[Tuple[str, str]] = None
    ) -> Dict[str, Any]:
        """对外接口：执行一次对话（集成LangSmith追踪）"""
        # 转换历史对话格式（Tuple→BaseMessage）
        parsed_history = []
        if chat_history:
            for human_msg, ai_msg in chat_history:
                parsed_history.append(HumanMessage(content=human_msg))
                parsed_history.append(AIMessage(content=ai_msg))

        # 判断是否需要执行索引
        need_index = bool(file_paths and len(file_paths) > 0)
        
        # 如果需要索引，先执行索引
        if need_index:
            logger.info(f"正在执行首次索引：{len(file_paths)}个文件")
            doc_ids = self.indexing_chain.run(file_paths)
            logger.info(f"首次索引完成：{len(doc_ids)}个文档片段存入向量库")
            indexing_completed = True
            initial_documents = [f"已索引文档ID: {id}" for id in doc_ids[:3]]
        else:
            indexing_completed = True
            initial_documents = []

        # 构造初始状态
        initial_state = AgentState(
            query=query,
            file_paths=file_paths or [],
            chat_history=parsed_history,
            indexing_completed=indexing_completed,
            documents=initial_documents
        )

        # 创建LangSmith追踪配置
        config = RunnableConfig(
            callbacks=[self.tracer] if self.tracer else None,
            tags=["rag-agent", "production"],
            metadata={
                "query": query,
                "has_files": bool(file_paths),
                "history_length": len(chat_history) if chat_history else 0,
                "need_index": need_index
            }
        )
        
        try:
            # 执行流程（带追踪）
            final_state = self.graph.invoke(initial_state, config=config)
            
            # 记录成功追踪
            if self.langsmith_client and self.tracer:
                run_id = getattr(self.tracer, 'run_id', 'unknown')
                logger.info(f"LangSmith追踪ID: {run_id}")
                
        except Exception as e:
            # 记录错误追踪
            if self.langsmith_client and self.tracer:
                try:
                    run_id = getattr(self.tracer, 'run_id', None)
                    if run_id and run_id != 'unknown':
                        self.langsmith_client.create_feedback(
                            run_id=run_id,
                            key="error",
                            score=0.0,
                            comment=str(e)
                        )
                    else:
                        logger.warning(f"无法记录错误追踪，run_id无效: {run_id}")
                except Exception as feedback_error:
                    logger.warning(f"记录错误追踪失败: {feedback_error}")
            logger.error(f"RAG流程执行失败: {str(e)}")
            raise

        # 提取最终状态数据
        try:
            final_query = final_state.get("query", query) if hasattr(final_state, 'get') else getattr(final_state, "query", query)
        except:
            final_query = query
            
        try:
            final_reformulated_query = final_state.get("reformulated_query", "") if hasattr(final_state, 'get') else getattr(final_state, "reformulated_query", "")
        except:
            final_reformulated_query = ""
            
        try:
            final_answer = final_state.get("answer", "") if hasattr(final_state, 'get') else getattr(final_state, "answer", "")
        except:
            final_answer = ""
            
        try:
            final_documents = final_state.get("documents", []) if hasattr(final_state, 'get') else getattr(final_state, "documents", [])
        except:
            final_documents = []
            
        try:
            final_indexing_completed = final_state.get("indexing_completed", indexing_completed) if hasattr(final_state, 'get') else getattr(final_state, "indexing_completed", indexing_completed)
        except:
            final_indexing_completed = indexing_completed
        
        # 提取聊天历史
        try:
            final_chat_history = final_state.get("chat_history", parsed_history) if hasattr(final_state, 'get') else getattr(final_state, "chat_history", parsed_history)
        except:
            final_chat_history = parsed_history
        
        # 如果 chat_history 是 BaseMessage 对象列表，需要转换为元组格式
        formatted_history = []
        if final_chat_history:
            # 确保 chat_history 是列表格式
            if hasattr(final_chat_history, '__iter__') and not isinstance(final_chat_history, str):
                chat_list = list(final_chat_history)
                # 成对处理聊天记录
                for i in range(0, len(chat_list), 2):
                    if i + 1 < len(chat_list):
                        human_msg = chat_list[i]
                        ai_msg = chat_list[i + 1]
                        # 获取消息内容
                        human_content = human_msg.content if hasattr(human_msg, 'content') else str(human_msg)
                        ai_content = ai_msg.content if hasattr(ai_msg, 'content') else str(ai_msg)
                        formatted_history.append((human_content, ai_content))

        return {
            "query": final_query,
            "reformulated_query": final_reformulated_query,
            "answer": final_answer,
            "retrieved_doc_count": len(final_documents),
            "retrieved_documents": final_documents,  
            "chat_history": formatted_history,
            "indexing_completed": final_indexing_completed
        }