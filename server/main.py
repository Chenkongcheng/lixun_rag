import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_agent import RAGAgent
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_interactive")


def interactive_conversation():
    """交互式对话"""

    # 配置需要解析的文档
    INITIAL_FILES = [
        "E:/五，RAG项目实战企业篇/课件/fufan-chat-api-4.0.0/fufan-chat-api-4.0.0/requirements.txt",
    ]
    COLLECTION_NAME = "interactive_rag_collection"

    rag_agent = RAGAgent(collection_name=COLLECTION_NAME)
    chat_history = None  
    is_first_round = True  

    try:
        while True:
            user_query = input("\n请输入你的问题（输入「退出」结束）：").strip()
    
            if user_query.lower() == "退出":
                logger.info("会话结束，正在清理资源...")
                break
        
            if not user_query:
                logger.warning("请输入有效问题，不要为空！")
                continue

            logger.info("正在处理您的问题...")
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

                print("\n" + "-"*50)
                print(f"你的问题：{result['query']}")
                if not is_first_round:
                    print(f"改写后问题：{result['reformulated_query']}")
                print(f"助手回答：\n{result['answer']}")
                print(f"检索到相关文档数：{result['retrieved_doc_count']}")
                print("-"*50)
                
                # 使用日志记录详细信息
                logger.info(f"对话完成 - 问题: {result['query']}, 改写: {result['reformulated_query']}, 文档数: {result['retrieved_doc_count']}")

                # 更新历史对话（供下一轮使用）
                chat_history = result["chat_history"]

            except Exception as e:
                logger.error(f"处理失败：{str(e)}")
                logger.error(f"对话错误：{str(e)}", exc_info=True)

    finally:
        # 清理资源
        rag_agent.vector_store.delete_collection()
        logger.info("资源清理完成，再见！")


if __name__ == "__main__":
    interactive_conversation()