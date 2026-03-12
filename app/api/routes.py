import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

from app.core.config import config
from app.models.schemas import QueryRequest, QueryResponse, DocumentIngestRequest, HealthResponse
from app.rag.engine import EnterpriseRAGEngine

logger = logging.getLogger(__name__)

# 创建 API 路由器
router = APIRouter()

# 实例化全局 RAG 引擎（会在 FastAPI 启动时自动装载所有的模型和数据库）
try:
    rag_engine = EnterpriseRAGEngine()
except Exception as e:
    logger.critical(f"RAG 引擎初始化失败，请检查配置和依赖: {e}")
    rag_engine = None


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """系统健康检查接口"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="RAG 引擎未就绪")

    return HealthResponse(
        status="ok",
        components={
            "vectorstore (Chroma)": "ready" if rag_engine.vectorstore else "error",
            "keyword_search (Elasticsearch)": "ready" if rag_engine.es_client else "not_configured_or_down",
            "cache (Redis)": "connected" if rag_engine.cache.redis else "disabled",
            "llm": "ready"
        }
    )


@router.post("/v1/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """核心问答接口"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="服务未就绪")

    try:
        # 执行端到端查询
        response = rag_engine.query(request)
        return response
    except Exception as e:
        logger.error(f"查询处理异常: {e}")
        raise HTTPException(status_code=500, detail=f"内部查询错误: {str(e)}")


@router.post("/v1/ingest")
async def ingest_documents(request: DocumentIngestRequest, background_tasks: BackgroundTasks):
    """文档入库接口（异步后台处理）"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="服务未就绪")

    def process_task():
        try:
            rag_engine.ingest_documents(request)
        except Exception as e:
            logger.error(f"后台文档入库任务失败: {e}")

    # 将耗时的文档解析入库任务丢给 FastAPI 的后台线程
    background_tasks.add_task(process_task)

    return {
        "status": "processing",
        "message": f"已接收 {len(request.file_paths)} 个文档的入库请求，正在后台排队处理中..."
    }


@router.delete("/v1/cache/{question}")
async def invalidate_cache(question: str, user_id: Optional[str] = None):
    """手动清除特定查询缓存接口"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="服务未就绪")

    rag_engine.cache.invalidate(question, user_id)
    return {"status": "success", "message": "缓存已清理"}