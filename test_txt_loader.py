#!/usr/bin/env python3
"""测试TXT加载器的编码兼容性"""

from rag.module.loader.txt_loader import CustomizedTxtLoader
import logging

logging.basicConfig(level=logging.INFO)

def test_txt_loader():
    """测试TXT加载器"""
    file_path = 'C:/Users/24109/Desktop/RAG测试集/凡人修仙传.txt'
    
    print(f"正在测试文件: {file_path}")
    
    try:
        # 创建加载器（不指定编码，让系统自动检测）
        loader = CustomizedTxtLoader(file_path)
        print(f"检测到的编码: {loader.file_encoding}")
        
        # 加载文档
        docs = loader.load()
        print(f"成功加载文档数: {len(docs)}")
        
        if docs:
            print(f"文档内容预览（前200字符）:")
            print(docs[0].page_content[:200])
            print("...")
            
        print("✅ TXT加载器测试成功！")
        
    except Exception as e:
        print(f"❌ TXT加载器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_txt_loader()