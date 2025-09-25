from abc import ABC, abstractmethod
from typing import List, Iterator
from langchain_core.documents import Document


class BaseLoader(ABC):
    """文档加载器的基础接口"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._check_file_exists()
    
    def _check_file_exists(self):
        """检查文件是否存在"""
        import os
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        if not os.path.isfile(self.file_path):
            raise ValueError(f"{self.file_path} 不是一个有效的文件")
    
    @abstractmethod
    def load(self) -> List[Document]:
        """加载文档并返回Document列表"""
        pass
    
    @abstractmethod
    def lazy_load(self) -> Iterator[Document]:
        """懒加载文档，逐个返回Document"""
        pass
    
    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> List[str]:
        """返回该加载器支持的文件扩展名列表"""
        pass