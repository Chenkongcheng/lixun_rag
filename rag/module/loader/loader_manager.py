from typing import List, Union, Iterator
from langchain_core.documents import Document
from .base_loader import BaseLoader
import os

from .pdf_loader import CustomizedOcrPdfLoader
from .doc_loader import CustomizedOcrDocLoader
from .md_loader import CustomizedMdLoader
from .txt_loader import CustomizedTxtLoader
from .web_loader import CustomizedWebLoader

import logging
logger = logging.getLogger(__name__)

class LoaderManager:
    """加载器管理器，根据文件类型自动选择合适的加载器"""
    
    def __init__(self):
        self._loaders = [
            CustomizedOcrPdfLoader,
            CustomizedOcrDocLoader,
            CustomizedMdLoader,
            CustomizedTxtLoader,
            CustomizedWebLoader
        ]
        self._extension_map = self._build_extension_map()

    def _build_extension_map(self) -> dict:
        """构建文件扩展名到加载器的映射"""
        extension_map = {}
        for loader_class in self._loaders:
            for ext in loader_class.supported_extensions():
                extension_map[ext.lower()] = loader_class
        return extension_map

    def get_loader(self, file_path: str) -> BaseLoader:
        """
        根据文件路径获取合适的加载器
        
        Args:
            file_path: 文件路径或URL
            
        Returns:
            对应的加载器实例
            
        Raises:
            ValueError: 不支持的文件类型
        """
        # 处理URL的情况
        if file_path.startswith(("http://", "https://")):
            for ext, loader_class in self._extension_map.items():
                if file_path.startswith(ext):
                    return loader_class(web_path=file_path)
            raise ValueError(f"不支持的URL类型: {file_path}")
        
        # 处理本地文件的情况
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in self._extension_map:
            return self._extension_map[ext](file_path=file_path)
        
        # 如果没有匹配的扩展名，尝试所有加载器
        for loader_class in self._loaders:
            if ext in loader_class.supported_extensions():
                return loader_class(file_path=file_path)
        
        raise ValueError(f"不支持的文件类型: {ext}，文件路径: {file_path}")

    def load(self, file_path: str) -> List[Document]:
        """
        加载单个文件
        
        Args:
            file_path: 文件路径或URL
            
        Returns:
            文档列表
        """
        loader = self.get_loader(file_path)
        return loader.load()

    def load_multiple(self, file_paths: List[str]) -> List[Document]:
        """
        加载多个文件
        
        Args:
            file_paths: 文件路径或URL列表
            
        Returns:
            所有文档的列表
        """
        all_docs = []
        for path in file_paths:
            try:
                docs = self.load(path)
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"加载文件 {path} 失败: {str(e)}")
        return all_docs

    def lazy_load(self, file_path: str) -> Iterator[Document]:
        """
        懒加载单个文件
        
        Args:
            file_path: 文件路径或URL
            
        Yields:
            文档
        """
        loader = self.get_loader(file_path)
        yield from loader.lazy_load()

    def get_supported_extensions(self) -> List[str]:
        """返回所有支持的文件扩展名"""
        return list(self._extension_map.keys())



loader_manager = LoaderManager()
