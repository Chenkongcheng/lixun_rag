"""
基本测试用例
"""
import unittest
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class TestBasicImports(unittest.TestCase):
    
    def test_base_classes(self):
        """测试基础类导入"""
        from rag.chains.base import BaseIndexingChain, BaseRetrievalChain, BaseGenerationChain
        self.assertTrue(hasattr(BaseIndexingChain, 'load'))
        self.assertTrue(hasattr(BaseRetrievalChain, 'retrieval'))
        self.assertTrue(hasattr(BaseGenerationChain, 'generate'))
    
    def test_config(self):
        """测试配置"""
        self.assertTrue('DASHSCOPE_API_KEY' in os.environ or 'LANGSMITH_API_KEY' in os.environ)

if __name__ == '__main__':
    unittest.main()