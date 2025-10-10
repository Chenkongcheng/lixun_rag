#!/bin/bash

# Lixun RAG 启动脚本
# 支持自动下载模型/向量库依赖，初始化服务

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker和Docker Compose
 check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi

    log_success "Docker环境检查通过"
}

# 检查环境变量
 check_env() {
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            log_warning "未找到.env文件，将使用.env.example作为模板"
            cp .env.example .env
            log_info "请编辑.env文件，配置您的API密钥"
            log_info "特别是 DASHSCOPE_API_KEY 必须配置"
        else
            log_error "未找到.env文件和.env.example文件"
            exit 1
        fi
    fi

    # 检查关键环境变量
    if grep -q "your-api-key-here" .env; then
        log_error "请先在.env文件中配置您的API密钥"
        log_info "将 DASHSCOPE_API_KEY 替换为您的实际API密钥"
        exit 1
    fi

    log_success "环境变量检查通过"
}

# 初始化数据目录
 init_directories() {
    log_info "初始化数据目录..."
    mkdir -p data logs documents ssl
    
    # 设置权限
    chmod 755 data logs documents ssl
    
    log_success "数据目录初始化完成"
}

# 下载模型和依赖
 download_models() {
    log_info "检查并下载必要的模型和依赖..."
    
    # 创建临时目录
    mkdir -p temp
    
    # 下载NLTK数据
    if [ ! -d "nltk_data" ]; then
        log_info "下载NLTK数据..."
        python3 -c "
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
" || log_warning "NLTK数据下载失败，将在容器中重新下载"
    fi
    
    # 清理临时文件
    rm -rf temp
    
    log_success "模型和依赖检查完成"
}

# 构建镜像
 build_images() {
    log_info "构建Docker镜像..."
    
    # 使用国内镜像源加速构建（可选）
    if [ "$USE_CN_MIRROR" = "true" ]; then
        export DOCKER_BUILDKIT=1
        docker build -t lixun-rag:latest --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
    else
        docker build -t lixun-rag:latest .
    fi
    
    log_success "镜像构建完成"
}

# 启动服务
 start_services() {
    log_info "启动服务..."
    
    # 根据环境选择不同的配置文件
    if [ "$ENV" = "production" ]; then
        log_info "使用生产环境配置"
        docker-compose --profile production up -d
    else
        log_info "使用开发环境配置"
        docker-compose up -d
    fi
    
    log_success "服务启动命令已发送"
}

# 等待服务就绪
 wait_for_services() {
    log_info "等待服务就绪..."
    
    # 等待主应用
    log_info "等待主应用就绪..."
    for i in {1..60}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log_success "主应用已就绪"
            break
        fi
        if [ $i -eq 60 ]; then
            log_error "主应用启动超时"
            exit 1
        fi
        sleep 2
    done
}

# 显示服务状态
 show_status() {
    log_info "服务状态："
    echo "=========================================="
    docker-compose ps
    echo "=========================================="
    
    log_info "服务访问地址："
    echo "- 主应用: http://localhost:8000"
    echo "- Redis缓存: localhost:6379"
    
    if [ "$ENV" = "production" ]; then
        echo "- Nginx代理: http://localhost"
    fi
    
    echo "=========================================="
    log_success "🎉 Lixun RAG 服务启动成功！"
}

# 清理函数
 cleanup() {
    log_info "正在清理..."
    docker-compose down
    log_success "清理完成"
}

# 主函数
 main() {
    log_info "🚀 开始启动 Lixun RAG 服务..."
    
    # 检查参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --production|-p)
                export ENV=production
                shift
                ;;
            --cn-mirror)
                export USE_CN_MIRROR=true
                shift
                ;;
            --clean)
                cleanup
                exit 0
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --production, -p    使用生产环境配置"
                echo "  --cn-mirror        使用国内镜像源加速构建"
                echo "  --clean            清理所有服务"
                echo "  --help, -h         显示帮助信息"
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                exit 1
                ;;
        esac
    done
    
    # 执行启动流程
    check_docker
    check_env
    init_directories
    download_models
    build_images
    start_services
    wait_for_services
    show_status
}

# 设置信号处理
trap cleanup EXIT

# 运行主函数
main "$@"