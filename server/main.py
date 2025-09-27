#!/usr/bin/env python3
"""
Lixun RAG 主服务入口
提供健康检查端点和交互式对话
"""
import os
import sys
import logging
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
import uvicorn
from health import app as health_app
from rag_agent import RAGAgent
from dotenv import load_dotenv

load_dotenv()

# 配置系统日志 - 用于记录系统运行状态和错误
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("rag_server.log", encoding='utf-8')  # 输出到文件
    ]
)

# 创建系统日志记录器
logger = logging.getLogger("rag_server")

# 配置控制台输出日志 - 专门用于用户交互显示
console_logger = logging.getLogger("console_output")
console_logger.setLevel(logging.INFO)
# 避免传播到根logger，防止重复输出
console_logger.propagate = False
# 添加控制台处理器
if not console_logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_logger.addHandler(console_handler)

# 定义生命周期管理器:cite[2]:cite[8]
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动和关闭事件处理:cite[1]:cite[8]"""
    logger.info("Lixun RAG 服务启动中...")
    logger.info("服务启动完成")
    yield  
    
    logger.info("Lixun RAG 服务关闭中...")
    logger.info("服务关闭完成")

# 创建主应用并传入lifespan:cite[1]:cite[8]
app = FastAPI(
    title="Lixun RAG API", 
    version="1.0.0",
    lifespan=lifespan  
)

# 挂载健康检查应用
app.mount("/", health_app)

def main():
    """主函数"""
    logger.info("🚀 启动 Lixun RAG 服务...")
    
    # 配置需要解析的文档
    INITIAL_FILES = [
        "E:/五，RAG项目实战企业篇/课件/fufan-chat-api-4.0.0/fufan-chat-api-4.0.0/requirements.txt",
    ]
    COLLECTION_NAME = "interactive_rag_collection"

    # 使用专门的logger输出到控制台，供用户查看
    console_logger.info("🚀 启动 Lixun RAG 服务...")
    console_logger.info("📝 正在初始化向量数据库和语言模型...")
    
    rag_agent = RAGAgent(collection_name=COLLECTION_NAME)
    chat_history = None  
    is_first_round = True  

    try:
        while True:
            # 使用专门的logger输出到控制台，供用户查看
            console_logger.info("\n请输入你的问题（输入「退出」结束）：")
            user_query = input().strip()
    
            if user_query.lower() == "退出":
                logger.info("会话结束，正在清理资源...")
                console_logger.info("👋 会话结束，正在清理资源...")
                break
        
            if not user_query:
                logger.warning("请输入有效问题，不要为空！")
                console_logger.info("⚠️ 请输入有效问题，不要为空！")
                continue

            logger.info("正在处理您的问题...")
            console_logger.info("🔄 正在处理您的问题...")
            try:
                if is_first_round:
                    result = rag_agent.run(
                        query=user_query,
                        file_paths=INITIAL_FILES,
                        chat_history=chat_history
                    )
                    is_first_round = False  
                else:
                    result = rag_agent.run(
                        query=user_query,
                        file_paths=None,  
                        chat_history=chat_history
                    )
                
                # 使用专门的logger输出到控制台，供用户查看
                console_logger.info("\n" + "="*60)
                console_logger.info(f"🤔 你的问题：{result['query']}")
                if result['reformulated_query'] != result['query']:
                    console_logger.info(f"🔧 改写后问题：{result['reformulated_query']}")
                console_logger.info(f"🤖 助手回答：\n{result['answer']}")
                console_logger.info(f"📚 参考文档数：{result['retrieved_doc_count']}个")
                
                # 获取并显示检索到的文档片段
                if 'retrieved_documents' in result and result['retrieved_documents']:
                    console_logger.info("\n📄 检索到的文档片段：")
                    for i, doc in enumerate(result['retrieved_documents'], 1):
                        # 获取文档内容
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
                        
                        # 显示文档片段（限制长度以避免过多输出）
                        max_content_length = 200
                        if len(content) > max_content_length:
                            content = content[:max_content_length] + "..."
                        
                        # 获取来源信息
                        source = metadata.get('source', '未知来源')
                        page = metadata.get('page', '')
                        page_info = f" (第{page}页)" if page else ""
                        
                        console_logger.info(f"  片段{i}: 来源={source}{page_info}")
                        console_logger.info(f"    内容: {content}")
                        console_logger.info("")
                console_logger.info("="*60)
                
                # 使用系统logger记录日志
                logger.info(f"对话完成 - 问题: {result['query']}, 改写: {result['reformulated_query']}, 文档数: {result['retrieved_doc_count']}")

                # 更新历史对话（供下一轮使用）
                chat_history = result["chat_history"]

            except Exception as e:
                logger.error(f"处理失败：{str(e)}")
                console_logger.error(f"❌ 处理失败：{str(e)}")
                logger.error(f"对话错误：{str(e)}", exc_info=True)

    finally:
        rag_agent.vector_store.delete_collection()
        logger.info("资源清理完成，再见！")
        console_logger.info("🧹 资源清理完成，再见！")

if __name__ == "__main__":
    # 如果是直接运行，启动HTTP服务
    # uvicorn.run(
    #     app, 
    #     host="0.0.0.0", 
    #     port=8000,
    #     log_config=None  # 使用我们自己的日志配置
    # )
    main()