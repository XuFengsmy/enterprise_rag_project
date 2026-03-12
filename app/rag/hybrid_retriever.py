import logging
from typing import List, Tuple, Dict, Any, Optional

from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class EnterpriseHybridRetriever:
    """企业级双路混合检索引擎：Chroma(向量语义) + Elasticsearch(BM25关键词) + RRF融合"""

    def __init__(self, vectorstore: Chroma, es_client: Elasticsearch, es_index_name: str = "enterprise_knowledge_base"):
        self.vectorstore = vectorstore
        self.es_client = es_client
        self.es_index_name = es_index_name
        self.rrf_k = 60  # RRF 平滑常数，标准值为 60

        # 测试 Elasticsearch 连接
        try:
            if self.es_client and self.es_client.ping():
                logger.info(f"✅ Elasticsearch 连接成功，目标索引: {self.es_index_name}")
            else:
                logger.warning("⚠️ Elasticsearch 连接失败！混合检索将自动降级为纯向量检索。")
                self.es_client = None
        except Exception as e:
            logger.error(f"❌ Elasticsearch 连接异常: {e}")
            self.es_client = None

    def _vector_search(self, query: str, top_k: int, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """单路召回一：向量检索 (ChromaDB)"""
        try:
            # 构建过滤参数
            search_kwargs = {"k": top_k}
            if metadata_filter:
                search_kwargs["filter"] = metadata_filter

            # Chroma 返回的是 Document 对象列表
            docs = self.vectorstore.similarity_search(query, **search_kwargs)
            return docs
        except Exception as e:
            logger.error(f"❌ 向量检索失败: {e}")
            return []

    def _keyword_search(self, query: str, top_k: int, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """单路召回二：关键词检索 (Elasticsearch BM25)"""
        if not self.es_client or not self.es_client.indices.exists(index=self.es_index_name):
            return []

        try:
            # 构建 ES 的 Query DSL
            es_query = {
                "bool": {
                    "must": [
                        {"match": {"content": query}}
                    ]
                }
            }

            # 处理元数据权限过滤（例如部门隔离、文件类型隔离）
            if metadata_filter:
                filter_clauses = []
                for key, value in metadata_filter.items():
                    # 在存入 ES 时，我们将 metadata 存在 metadata 字段下
                    filter_clauses.append({"term": {f"metadata.{key}.keyword": value}})
                es_query["bool"]["filter"] = filter_clauses

            # 执行查询
            response = self.es_client.search(
                index=self.es_index_name,
                query=es_query,
                size=top_k
            )

            # 解析 ES 的命中结果转换为统一的 LangChain Document 对象
            docs = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                docs.append(Document(
                    page_content=source.get("content", ""),
                    metadata=source.get("metadata", {})
                ))
            return docs
        except Exception as e:
            logger.error(f"❌ Elasticsearch 检索失败: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        核心方法：混合检索与 RRF 融合
        """
        # 【新增修复】清洗过滤条件，拦截 Swagger UI 传来的类似 {'additionalProp1': {}} 的脏数据
        if metadata_filter:
            # 只保留值不是空字典的键值对
            metadata_filter = {k: v for k, v in metadata_filter.items() if v != {}}
            # 如果清洗后变成了空字典，直接置为 None
            if not metadata_filter:
                metadata_filter = None
        # 为了保证重排序的质量，我们在单路召回时适当放宽数量 (通常是 top_k 的 2 倍)
        recall_k = top_k * 2

        logger.info(f"🔍 开始双路混合检索: '{query}' | 过滤条件: {metadata_filter}")

        # 1. 并发/串行获取两路召回结果
        vector_docs = self._vector_search(query, recall_k, metadata_filter)
        keyword_docs = self._keyword_search(query, recall_k, metadata_filter)

        # 2. RRF (Reciprocal Rank Fusion) 倒数秩融合算法
        rrf_scores = {}
        doc_map = {}  # 用于通过内容哈希映射回原始 Document 对象

        # 处理第一路：向量召回排名
        for rank, doc in enumerate(vector_docs):
            doc_key = hash(doc.page_content) # 使用内容哈希作为唯一标识
            doc_map[doc_key] = doc
            # RRF 公式: score = 1 / (rank + k)
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (rank + 1 + self.rrf_k)

        # 处理第二路：关键词召回排名
        for rank, doc in enumerate(keyword_docs):
            doc_key = hash(doc.page_content)
            doc_map[doc_key] = doc
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0.0) + 1.0 / (rank + 1 + self.rrf_k)

        # 3. 根据 RRF 融合分数降序排列
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. 组装最终输出
        final_results = []
        for doc_key, score in sorted_results[:top_k]:
            final_results.append((doc_map[doc_key], score))

        logger.info(f"🎯 混合检索融合完成！(向量命中 {len(vector_docs)} 条，关键词命中 {len(keyword_docs)} 条)")
        return final_results