"""
LangSmith可观测性配置
"""
import os
import sys
import logging
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from dotenv import load_dotenv
load_dotenv()  

# 设置sys.stdout的编码为UTF-8
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

# 配置日志输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('langsmith.log', encoding='utf-8')  # 设置UTF-8编码
    ]
)

logger = logging.getLogger(__name__)

def setup_langsmith():
    """设置LangSmith追踪"""
    try:
        # 调试：打印环境变量状态
        tracing_enabled = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        api_key = os.getenv("LANGSMITH_API_KEY")
        
        logger.info(f"LANGSMITH_TRACING: {tracing_enabled}")
        logger.info(f"LANGSMITH_API_KEY 存在: {api_key is not None}")
        
        if tracing_enabled:
            endpoint = os.getenv("LANGSMITH_ENDPOINT")
            project = os.getenv("LANGSMITH_PROJECT", "lixun-rag")

            if not api_key:
                logger.warning("LANGSMITH_API_KEY未设置，LangSmith追踪已禁用")
                return None

            # 创建LangSmith客户端
            client = Client(
                api_url=endpoint,
                api_key=api_key,
            )

            tracer = LangChainTracer(
                project_name=project,
                client=client,
            )

            logger.info(f"LangSmith追踪已启用 - 项目: {project}")
            return tracer
        else:
            logger.info("LangSmith追踪已禁用（LANGSMITH_TRACING不为true）")
            return None
    except Exception as e:
        logger.error(f"LangSmith设置失败: {str(e)}")
        return None

if __name__ == "__main__":
    # 测试时设置环境变量
    os.environ["LANGSMITH_TRACING"] = "true"
    # os.environ["LANGSMITH_API_KEY"] = "你的API密钥"  # 取消注释并填入真实密钥
    
    setup_langsmith()