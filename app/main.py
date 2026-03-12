import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

# 配置全局的基础日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 初始化 FastAPI 应用
app = FastAPI(
    title="🏢 企业级 RAG 知识库 API",
    description="支持混合检索、多步查询分解、重排序与权限控制的智能问答系统",
    version="1.0.0"
)

# 配置 CORS 跨域资源共享（生产环境建议收紧 allow_origins）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载 API 路由
app.include_router(router)


if __name__ == "__main__":
    # 启动 Uvicorn 服务器
    # 注意模块导入路径，因为当前文件在 app 目录下，所以是 app.main:app
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 开发模式下开启热重载
        workers=1     # Windows 下开发建议 1 个 worker 避免多进程冲突
    )