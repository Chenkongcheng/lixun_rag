# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装uv包管理器
RUN pip install --no-cache-dir uv

# 复制依赖文件
COPY pyproject.toml uv.lock ./
COPY nltk_data/ ./nltk_data/

# 安装Python依赖
RUN uv pip install --system --no-cache-dir -e .

# 复制应用代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data /app/logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV NLTK_DATA=/app/nltk_data

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["python", "server/main.py"]