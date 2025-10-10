import os
from typing import List, Optional, Any, Dict, Iterator
from dotenv import load_dotenv
from openai import OpenAI
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk


load_dotenv()

DASHSCOPE_API_KEY_FROM_ENV = os.getenv("DASHSCOPE_API_KEY")
ALIYUN_BASE_URL_FROM_ENV = os.getenv("ALIYUN_BASE_URL")
MODEL_NAME_FROM_ENV = os.getenv("LLM_MODEL_NAME")

class QwenLLM(LLM):
    """
    阿里云通义千问模型
    """

    api_key: Optional[str] = DASHSCOPE_API_KEY_FROM_ENV  
    
    # 模型配置参数
    model_name: str = MODEL_NAME_FROM_ENV
    temperature: float = 0.8
    base_url: str = ALIYUN_BASE_URL_FROM_ENV


    def __init__(self, **kwargs: Any):
        super().__init__(** kwargs)
        
        # 校验api_key
        if self.api_key is None or self.api_key.strip() == "":
            raise ValueError(
                "请提供DASHSCOPE_API_KEY！可选方式：\n"
                "1. 在实例化QwenLLM时传入参数 api_key='你的密钥'\n"
                "2. 在.env文件中配置 DASHSCOPE_API_KEY='你的密钥'"
            )


    @property
    def _client(self) -> OpenAI:
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )


    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                stop=stop,** kwargs
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            raise RuntimeError(f"通义千问API调用失败: {str(e)}") from e


    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """
        修复流式调用
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                stop=stop,
                stream=True,** kwargs
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    generation_chunk = GenerationChunk(text=token)
                    yield generation_chunk
                    
                    if run_manager:
                        run_manager.on_llm_new_token(token)
                        
        except Exception as e:
            raise RuntimeError(f"通义千问流式API调用失败: {str(e)}") from e


    @property
    def _llm_type(self) -> str:
        return "qwen"


    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "base_url": self.base_url,
            "api_key": "******"  
        }


