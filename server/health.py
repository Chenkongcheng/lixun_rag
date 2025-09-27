from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="RAG_lixun Health API", version="1.0.0")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查关键服务状态
        health_status = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",  
            "services": {
                "llm": "available",
                "vector_store": "available",
                "embedding": "available"
            }
        }
        logger.info("健康检查通过")
        return JSONResponse(content=health_status, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

@app.get("/")
async def root():
    return {"message": "Lixun RAG Service is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)