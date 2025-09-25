# 🔍 Lixun RAG - 智能文档问答系统

一个基于LangChain和Chroma向量数据库的RAG系统，支持多种文档格式的智能问答。

## 🚀 安装与启动指南

### 环境要求
- Python 3.10+
- Docker & Docker Compose


### 快速启动
```bash
# 1. 克隆项目
git clone https://github.com/Chenkongcheng/lixun_rag.git
cd RAG_lixun

# 2. 配置API密钥
cp .env.example .env
# 编辑.env文件，填写DASHSCOPE_API_KEY

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

## 🛠️ 技术栈说明

### 核心框架
- **LangChain**: RAG流程编排框架
- **LangGraph**: 智能体框架
- **Chroma**: 向量数据库存储文档嵌入
- **阿里通义千问**: 大语言模型提供问答能力


## 💻 示例使用命令

### 交互式对话
```bash
cd server
python main.py
```

### Docker操作
```bash
# 构建镜像
docker build -t lixun-rag .

# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```


## 📁 项目结构解释

```
lixun-rag/
├── 📁 server/          # 服务器端代码
│   ├── main.py        # 交互式对话主程序
│   └── rag_agent.py   # RAG代理核心实现
├── 📁 rag/            # RAG核心模块
│   ├── 📁 chains/     # 处理链（索引、检索、生成）
│   ├── 📁 connector/  # 连接器（LLM、向量数据库）
│   └── 📁 module/     # 功能模块（文档加载器）
├── 📁 scripts/        # 启动脚本
├── 📁 nltk_data/      # NLTK语言数据
├── 📄 Dockerfile      # 容器配置
├── 📄 docker-compose.yml  # 服务编排
├── 📄 pyproject.toml  # 项目配置
└── 📄 .env           # 环境变量
```

### 核心模块
- **chains/**: 文档索引、检索、生成的工作流
- **connector/**: 大模型和向量数据库的连接封装
- **module/loader/**: 各种格式文档的加载解析
- **server/**: 用户交互界面和业务逻辑

