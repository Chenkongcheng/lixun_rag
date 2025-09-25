@echo off
REM Lixun RAG Windows 启动脚本
REM 支持自动下载模型/向量库依赖，初始化服务

setlocal enabledelayedexpansion

REM 颜色定义
set "RED=[91m"
set "GREEN=[92m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM 日志函数
:log_info
echo %BLUE%[INFO]%NC% %~1%
goto :eof

:log_success
echo %GREEN%[SUCCESS]%NC% %~1%
goto :eof

:log_warning
echo %YELLOW%[WARNING]%NC% %~1%
goto :eof

:log_error
echo %RED%[ERROR]%NC% %~1%
goto :eof

REM 检查Docker
:check_docker
where docker >nul 2>nul
if %errorlevel% neq 0 (
    call :log_error "Docker未安装，请先安装Docker"
    exit /b 1
)

docker-compose --version >nul 2>nul
if %errorlevel% neq 0 (
    call :log_error "Docker Compose未安装，请先安装Docker Compose"
    exit /b 1
)

call :log_success "Docker环境检查通过"
goto :eof

REM 检查环境变量
:check_env
if not exist ".env" (
    if exist ".env.example" (
        call :log_warning "未找到.env文件，将使用.env.example作为模板"
        copy .env.example .env >nul
        call :log_info "请编辑.env文件，配置您的API密钥"
        call :log_info "特别是 DASHSCOPE_API_KEY 必须配置"
    ) else (
        call :log_error "未找到.env文件和.env.example文件"
        exit /b 1
    )
)

REM 检查关键环境变量
findstr /C:"your-api-key-here" .env >nul 2>nul
if %errorlevel% equ 0 (
    call :log_error "请先在.env文件中配置您的API密钥"
    call :log_info "将 DASHSCOPE_API_KEY 替换为您的实际API密钥"
    exit /b 1
)

call :log_success "环境变量检查通过"
goto :eof

REM 初始化目录
:init_directories
call :log_info "初始化数据目录..."
if not exist data mkdir data
if not exist logs mkdir logs
if not exist documents mkdir documents
if not exist ssl mkdir ssl

call :log_success "数据目录初始化完成"
goto :eof

REM 下载模型和依赖
:download_models
call :log_info "检查并下载必要的模型和依赖..."

REM 检查NLTK数据
if not exist nltk_data (
    call :log_info "下载NLTK数据..."
    python -c "
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
" 2>nul
    if %errorlevel% neq 0 (
        call :log_warning "NLTK数据下载失败，将在容器中重新下载"
    )
)

call :log_success "模型和依赖检查完成"
goto :eof

REM 构建镜像
:build_images
call :log_info "构建Docker镜像..."

if "%USE_CN_MIRROR%"=="true" (
    set DOCKER_BUILDKIT=1
    docker build -t lixun-rag:latest --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .
) else (
    docker build -t lixun-rag:latest .
)

if %errorlevel% neq 0 (
    call :log_error "镜像构建失败"
    exit /b 1
)

call :log_success "镜像构建完成"
goto :eof

REM 启动服务
:start_services
call :log_info "启动服务..."

if "%ENV%"=="production" (
    call :log_info "使用生产环境配置"
    docker-compose --profile production up -d
) else (
    call :log_info "使用开发环境配置"
    docker-compose up -d
)

if %errorlevel% neq 0 (
    call :log_error "服务启动失败"
    exit /b 1
)

call :log_success "服务启动命令已发送"
goto :eof

REM 等待服务就绪
:wait_for_services
call :log_info "等待服务就绪..."

REM 等待主应用
call :log_info "等待主应用就绪..."
set attempts=0
:wait_app
set /a attempts+=1
if %attempts% gtr 60 (
    call :log_error "主应用启动超时"
    exit /b 1
)
curl -f http://localhost:8000/health >nul 2>nul
if %errorlevel% neq 0 (
    timeout /t 2 /nobreak >nul
    goto wait_app
)
call :log_success "主应用已就绪"
goto :eof

REM 显示状态
:show_status
call :log_info "服务状态："
echo ==========================================
docker-compose ps
echo ==========================================

call :log_info "服务访问地址："
echo - 主应用: http://localhost:8000
echo - Redis缓存: localhost:6379

if "%ENV%"=="production" (
    echo - Nginx代理: http://localhost
)

echo ==========================================
call :log_success "🎉 Lixun RAG 服务启动成功！"
goto :eof

REM 清理函数
:cleanup
call :log_info "正在清理..."
docker-compose down
call :log_success "清理完成"
goto :eof

REM 主函数
:main
call :log_info "🚀 开始启动 Lixun RAG 服务..."

REM 解析参数
:parse_args
if "%1"=="" goto :args_done
if /i "%1"=="--production" (
    set ENV=production
    shift
    goto parse_args
)
if /i "%1"=="-p" (
    set ENV=production
    shift
    goto parse_args
)
if /i "%1"=="--cn-mirror" (
    set USE_CN_MIRROR=true
    shift
    goto parse_args
)
if /i "%1"=="--clean" (
    call :cleanup
    exit /b 0
)
if /i "%1"=="--help" (
    echo 用法: %0 [选项]
    echo 选项:
    echo   --production, -p    使用生产环境配置
    echo   --cn-mirror        使用国内镜像源加速构建
    echo   --clean            清理所有服务
    echo   --help             显示帮助信息
    exit /b 0
)
call :log_error "未知参数: %1"
exit /b 1

:args_done

REM 执行启动流程
call :check_docker
call :check_env
call :init_directories
call :download_models
call :build_images
call :start_services
call :wait_for_services
call :show_status

exit /b 0

REM 运行主函数
call :main %*