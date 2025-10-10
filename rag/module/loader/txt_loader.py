import os
import chardet
from typing import List, Iterator
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from .base_loader import BaseLoader
import logging

logger = logging.getLogger(__name__)


class CustomizedTxtLoader(TextLoader, BaseLoader):
    """TXT文档加载器"""
    
    def __init__(self, file_path: str, encoding: str = None):
        # 如果没有指定编码，自动检测文件编码
        if encoding is None:
            encoding = self._detect_encoding(file_path)
        
        TextLoader.__init__(self, file_path=file_path, encoding=encoding)
        BaseLoader.__init__(self, file_path=file_path)
        self.file_encoding = encoding

    def _detect_encoding(self, file_path: str) -> str:
        """检测文件编码"""
        try:
            # 读取文件的前几KB来检测编码
            with open(file_path, 'rb') as f:
                raw_data = f.read(10240)  # 读取前10KB
                
            # 使用chardet检测编码
            result = chardet.detect(raw_data)
            detected_encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            # 如果检测到的编码是GB2312或GBK，统一使用GBK
            if detected_encoding and detected_encoding.lower() in ['gb2312', 'gbk', 'gb18030']:
                detected_encoding = 'gbk'
            elif not detected_encoding or detected_encoding.lower() == 'ascii':
                detected_encoding = 'utf-8'
            
            # 验证检测到的编码是否真正能解码文件
            encoding_valid = False
            try:
                raw_data.decode(detected_encoding)
                encoding_valid = True
            except UnicodeDecodeError:
                encoding_valid = False
            
            # 如果检测到的编码无效或置信度较低，尝试更多编码检测
            if not encoding_valid or confidence < 0.8:
                # 尝试更多的中文编码
                chinese_encodings = ['utf-8', 'gbk', 'gb18030', 'big5']
                
                for encoding in chinese_encodings:
                    try:
                        decoded_text = raw_data.decode(encoding, errors='strict')
                        # 检查解码后的文本是否包含中文字符
                        if self._contains_chinese(decoded_text):
                            detected_encoding = encoding
                            encoding_valid = True
                            break
                    except UnicodeDecodeError:
                        continue
                
                # 如果严格解码失败，尝试使用错误处理
                if not encoding_valid:
                    for encoding in chinese_encodings + ['iso-8859-1']:
                        try:
                            # 使用ignore错误处理来测试编码
                            raw_data.decode(encoding, errors='ignore')
                            # 如果包含中文字符，优先选择
                            if encoding != 'iso-8859-1':
                                test_text = raw_data.decode(encoding, errors='ignore')
                                if self._contains_chinese(test_text):
                                    detected_encoding = encoding
                                    break
                            else:
                                detected_encoding = encoding
                                break
                        except:
                            continue
                
                # 如果仍然没有找到合适的编码，使用utf-8作为默认
                if not encoding_valid and detected_encoding == 'iso-8859-1':
                    detected_encoding = 'utf-8'
                
            logger.info(f"检测到文件 {file_path} 的编码为: {detected_encoding} (置信度: {confidence})")
            return detected_encoding
            
        except Exception as e:
            logger.warning(f"检测文件编码失败 {file_path}: {str(e)}，默认使用UTF-8")
            return 'utf-8'
    
    def _contains_chinese(self, text: str) -> bool:
        """检查文本是否包含中文字符"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 基本汉字范围
                return True
            if '\u3400' <= char <= '\u4dbf':  # 扩展A区
                return True
        return False

    def load(self) -> List[Document]:
        """加载文档"""
        try:
            return super().load()
        except (UnicodeDecodeError, RuntimeError) as e:
            logger.warning(f"使用编码 {self.file_encoding} 加载文件失败，尝试使用其他编码: {str(e)}")
            # 尝试多种备选编码
            fallback_encodings = ['gbk', 'gb18030', 'big5', 'utf-8', 'iso-8859-1']
            
            for encoding in fallback_encodings:
                if encoding != self.file_encoding.lower():
                    try:
                        logger.info(f"尝试使用编码 {encoding} 重新加载文件")
                        # 创建新的加载器实例
                        new_loader = TextLoader(file_path=self.file_path, encoding=encoding)
                        result = new_loader.load()
                        # 更新当前实例的编码
                        self.encoding = encoding
                        self.file_encoding = encoding
                        logger.info(f"成功使用编码 {encoding} 加载文件")
                        return result
                    except Exception:
                        continue
            
            # 如果所有编码都失败，使用错误处理模式
            logger.warning("所有编码尝试失败，使用iso-8859-1编码和错误忽略模式")
            try:
                # 使用iso-8859-1编码，忽略解码错误
                with open(self.file_path, 'r', encoding='iso-8859-1', errors='ignore') as f:
                    text = f.read()
                from langchain_core.documents import Document
                return [Document(page_content=text, metadata={"source": self.file_path})]
            except Exception as final_e:
                logger.error(f"最终尝试也失败: {str(final_e)}")
                raise RuntimeError(f"无法加载文件 {self.file_path}，尝试了多种编码") from final_e

    def lazy_load(self) -> Iterator[Document]:
        """懒加载文档"""
        try:
            for doc in super().lazy_load():
                yield doc
        except (UnicodeDecodeError, RuntimeError) as e:
            logger.warning(f"使用编码 {self.file_encoding} 懒加载文件失败，尝试使用其他编码: {str(e)}")
            # 尝试多种备选编码
            fallback_encodings = ['gbk', 'gb18030', 'big5', 'utf-8', 'iso-8859-1']
            
            for encoding in fallback_encodings:
                if encoding != self.file_encoding.lower():
                    try:
                        logger.info(f"尝试使用编码 {encoding} 重新懒加载文件")
                        # 创建新的加载器实例
                        new_loader = TextLoader(file_path=self.file_path, encoding=encoding)
                        for doc in new_loader.lazy_load():
                            yield doc
                        # 更新当前实例的编码
                        self.encoding = encoding
                        self.file_encoding = encoding
                        return
                    except Exception:
                        continue
            
            # 如果所有编码都失败，使用错误处理模式
            logger.warning("所有编码尝试失败，使用iso-8859-1编码和错误忽略模式")
            try:
                # 使用iso-8859-1编码，忽略解码错误
                with open(self.file_path, 'r', encoding='iso-8859-1', errors='ignore') as f:
                    text = f.read()
                from langchain_core.documents import Document
                yield Document(page_content=text, metadata={"source": self.file_path})
            except Exception as final_e:
                logger.error(f"最终尝试也失败: {str(final_e)}")
                raise RuntimeError(f"无法懒加载文件 {self.file_path}，尝试了多种编码") from final_e

    @classmethod
    def supported_extensions(cls) -> List[str]:
        """支持的文件扩展名"""
        return [".txt"]

    