import os
from pathlib import Path
from dotenv import load_dotenv

# 加载根目录的 .env 文件
BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    """集中管理所有配置"""
    # HuggingFace 配置
    HUGGINGFACEHUB_API_TOKEN = os.getenv("HF_READ_API")

    # 远程大模型 (调用 HF Inference API)
    LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    # 本地向量模型 (彻底告别 API 限制，支持本地 CPU/GPU 高速无限次并行计算)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # 路径与连接配置
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", str(BASE_DIR / "data" / "chroma_db"))
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    ES_URL = os.getenv("ES_URL", "http://enterprise_rag_es:9200")

    # RAG 参数配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "5"))

    # 功能开关
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
    QUERY_DECOMPOSITION_ENABLED = os.getenv("QUERY_DECOMPOSITION_ENABLED", "true").lower() == "true"


config = Config()