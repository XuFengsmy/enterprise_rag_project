from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """用户查询请求体"""
    question: str = Field(..., description="用户问题")
    user_id: Optional[str] = Field(None, description="用户ID，用于缓存隔离和权限过滤")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="元数据过滤条件")
    stream: bool = Field(False, description="是否流式输出")
    top_k: Optional[int] = Field(None, description="返回文档数量")

class QueryResponse(BaseModel):
    """查询响应体"""
    answer: str = Field(..., description="LLM 生成的最终答案")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="引用的溯源文档列表")
    confidence: float = Field(0.0, description="检索最高置信度分数")
    processing_time: float = Field(0.0, description="整体处理耗时（秒）")
    method: str = Field("direct", description="处理方式 (direct/decomposed)")

class DocumentIngestRequest(BaseModel):
    """文档入库请求体"""
    file_paths: List[str] = Field(..., description="本地需要处理的文档绝对路径列表")
    metadata: Optional[Dict[str, Any]] = Field(None, description="全局元数据（如归属部门、文档类型等）")

class HealthResponse(BaseModel):
    """健康检查响应体"""
    status: str = Field(..., description="服务整体状态")
    version: str = Field("1.0.0", description="API版本号")
    components: Dict[str, str] = Field(..., description="各个底层组件的状态 (LLM, Redis, 向量库等)")