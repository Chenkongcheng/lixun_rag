# 🔍 Lixun RAG - 智能文档问答系统

一个基于 LangChain 和 Chroma 向量数据库的 RAG 系统，支持多种文档格式的智能问答和多轮对话。

## 🚀 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose
- 阿里云 DASHSCOPE API 密钥

### 一键启动

```bash
# 1. 克隆项目
git clone https://github.com/Chenkongcheng/lixun_rag.git
cd RAG_lixun

# 2. 配置环境变量
cp .env
# 编辑.env文件，配置您的API密钥

# 3.配置要解析的文档路径：在server/main.py下配置文档路径，支持多个文档同时解析
INITIAL_FILES = [
        "E:/requirements.txt",
        "D:/TEST.pdf",
        ....
    ]

# 4. 一键启动
./scripts/start.sh        # Linux/Mac
scripts\start.bat         # Windows
```

### 手动安装

```bash
# 安装依赖
pip install uv
uv pip install -e .

# 启动服务
python server/main.py
```

## 📋 项目规范

### PEP 8 命名规范

本项目严格遵守 Python PEP 8 命名规范：

- **模块文件名**：全小写，必要时使用下划线（如 `chroma_store.py`）
- **类名**：使用驼峰命名法（如 `RAGAgent`）
- **函数/方法名**：全小写，下划线分隔（如 `reformulate_query`）
- **常量名**：全大写，下划线分隔（如 `MAX_CHUNK_SIZE`）

### 目录结构

```plaintext
lixun_rag/                              # 项目根目录（小写+下划线）
├── 📁 rag/                            # RAG核心模块
│   ├── 📁 chains/                     # 处理链
│   │   ├── base.py                    # 抽象基类
│   │   ├── indexing.py                # 文档索引
│   │   ├── retrieval.py               # 向量检索
│   │   └── generate.py                # 答案生成
│   ├── 📁 connector/                  # 外部服务连接器
│   │   ├── 📁 embedding/
│   │   │   └── aliyun.py              # 阿里云嵌入模型
│   │   ├── 📁 llm/
│   │   │   └── aliyun.py              # 阿里云大模型
│   │   └── 📁 vectorstore/
│   │       └── chroma_store.py        # Chroma向量数据库
│   └── 📁 module/                     # 功能模块
│       └── 📁 loader/                 # 文档加载器
├── 📁 server/                         # 服务器端
│   ├── main.py                        # FastAPI主服务
│   ├── rag_agent.py                   # RAG代理核心
│   └── langsmith_setup.py             # LangSmith配置
├── 📁 tests/                          # 测试用例
│   └── test_basic.py                  # 基础测试
├── 📁 scripts/                        # 启动脚本
├── 📁 data/                           # 数据存储
```

## 🔍 LangSmith 可观测性指南

### 什么是 LangSmith？

LangSmith 是 LangChain 官方提供的可观测性平台，用于追踪、监控和调试 RAG 应用。

### 配置步骤

1. **注册 LangSmith 账户**
   - 访问 LangSmith 官网
   - 创建账户并获取 API 密钥

2. **配置环境变量**
   
   在 `.env` 文件中添加：
   
   ```bash
   # LangSmith 配置
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your_langsmith_api_key_here
   LANGSMITH_PROJECT=lixun-rag-production
   ```

3. **查看追踪数据**
   - 启动应用并进行对话
   - 登录 LangSmith 平台查看完整的请求链路
   - 分析每个节点的执行时间、输入输出

### 追踪内容

✅ 查询改写过程  
✅ 文档检索结果  
✅ 答案生成流程  
✅ 错误和异常信息  
✅ 性能指标（延迟、token 用量）

## 🛠️ 技术栈

### 核心框架
- **LangChain**: RAG流程编排框架
- **LangGraph**: 智能体框架
- **Chroma**: 向量数据库存储文档嵌入
- **阿里通义千问**: 大语言模型提供问答能力


### AI 服务

- **阿里通义千问**：大语言模型（问答生成）
- **通义 Embedding**: 文本向量化模型
- **LangSmith**: 可观测性平台

### 开发工具

- **uv**: 快速的 Python 包管理器
- **pytest**: 单元测试框架
- **Docker**: 容器化部署

## 💻 使用示例

### 交互式对话

```bash
cd server
python main.py

# 输出示例：
# 请输入你的问题：什么是RAG？
# 助手回答：RAG是检索增强生成的技术...
```

### API 服务

```bash
# 启动HTTP服务
python server/main.py

# 健康检查
curl http://localhost:8000/health

# 对话API（需实现对应端点）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是机器学习？"}'
```

### Docker 部署

```bash
# 构建镜像
docker build -t lixun-rag:latest .

# 启动服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f app

# 停止服务
docker-compose down
```

## 🔧 开发指南

### 运行测试

```bash
# 运行基础测试
cd tests
python -m unittest test_basic.py

# 运行所有测试
python -m unittest discover
```

## 🗄️ Chroma 向量数据库配置

### 持久化目录配置

Chroma 向量数据库使用 `VECTOR_DB_PATH` 环境变量来指定持久化目录。默认情况下，系统会将向量数据存储在 `./data/chroma` 目录下。

```bash
# 在 .env 文件中配置向量数据库路径
VECTOR_DB_PATH=./data/chroma
```

### 持久化目录结构

Chroma 会在指定的持久化目录下创建以下结构：

```plaintext
data/chroma/                          # 持久化根目录
├── chroma.sqlite3                    # SQLite 数据库文件
├── {uuid}/                           # UUID 命名的集合目录
│   ├── data/                         # 向量数据
│   └── metadata.json                 # 元数据文件
└── {collection_name}/                # 集合特定目录
    ├── data/                         # 向量数据
    └── metadata.json                 # 元数据文件
```

### 环境变量一致性说明

系统确保 Chroma 的持久化目录（`persist_directory`）与 `VECTOR_DB_PATH` 环境变量保持一致：

1. **初始化过程**：
   - `RAGAgent` 类从环境变量 `VECTOR_DB_PATH` 读取持久化路径
   - 如果未设置，则使用默认值 `./data/chroma`
   - 路径会传递给 `ChromaVectorStore` 作为持久化目录

2. **路径一致性**：
   - 系统会对路径进行标准化处理，确保不同格式（相对路径、绝对路径）的一致性
   - 支持大小写不敏感的比较（Windows 系统）
   - 自动处理路径分隔符（`/` 和 `\`）

3. **Docker 部署**：
   - 在 Docker 环境中，建议使用挂载卷来持久化向量数据
   - 确保 `VECTOR_DB_PATH` 指向挂载卷内的路径

```yaml
# docker-compose.yml 示例
services:
  app:
    volumes:
      - ./data:/app/data  # 挂载数据目录
    environment:
      - VECTOR_DB_PATH=/app/data/chroma  # 使用容器内路径
```

### 多实例数据共享

通过正确配置 `VECTOR_DB_PATH`，多个 RAGAgent 实例可以共享同一个向量数据库：

```python
# 实例1
agent1 = RAGAgent(vector_db_path="./data/chroma")

# 实例2 - 使用相同的路径，共享数据
agent2 = RAGAgent(vector_db_path="./data/chroma")
```



