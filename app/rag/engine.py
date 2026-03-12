import os

from langchain_community.vectorstores.utils import filter_complex_metadata
import time
import logging
from typing import List, Dict, Any

from elasticsearch import Elasticsearch
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

from app.core.config import config
from app.models.schemas import QueryRequest, QueryResponse, DocumentIngestRequest
from app.services.cache_manager import CacheManager
from app.rag.document_processor import EnterpriseDocumentProcessor
from app.rag.hybrid_retriever import EnterpriseHybridRetriever
from app.rag.query_decomposer import QueryDecomposer, MultiStepRetriever
from app.rag.reranker import EnterpriseReranker

# 设置日志
logger = logging.getLogger(__name__)


class EnterpriseRAGEngine:
    """企业级 RAG 核心引擎：总控枢纽，串联文档处理、双路召回、拆解、重排序与大模型"""

    def __init__(self):
        logger.info("🚀 正在启动企业级 RAG 核心引擎...")

        # 1. 初始化基础服务
        self.cache = CacheManager(config.REDIS_URL, config.CACHE_TTL)

        # 2. 初始化本地向量模型 (Local Embeddings)
        # 你的机器有多快，它切块入库就有多快，再也没有 API 的 Batch Size 限制！
        logger.info(f"⏳ 正在加载本地向量模型: {config.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # 若有N卡并配好了CUDA环境，可改为 'cuda'
            encode_kwargs={'normalize_embeddings': True}
        )

        # 3. 初始化远程大模型 (HuggingFace Inference Endpoint)
        logger.info(f"⏳ 正在连接 HuggingFace 远程模型: {config.LLM_MODEL}")
        if config.HUGGINGFACEHUB_API_TOKEN:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = config.HUGGINGFACEHUB_API_TOKEN

        llm_endpoint = HuggingFaceEndpoint(
            repo_id=config.LLM_MODEL,
            huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN,
            task="text-generation",
            temperature=0.01,
            max_new_tokens=1024,
            do_sample=True,
        )
        self.llm = ChatHuggingFace(llm=llm_endpoint)

        # 4. 初始化底层数据库 (Chroma & Elasticsearch)
        self.vectorstore = Chroma(
            persist_directory=config.VECTOR_STORE_PATH,
            embedding_function=self.embeddings,
            collection_name="enterprise_collection"
        )
        self.es_client = Elasticsearch(config.ES_URL) if config.ES_URL else None
        # 5. 初始化 RAG 组件
        self.processor = EnterpriseDocumentProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.retriever = EnterpriseHybridRetriever(
            vectorstore=self.vectorstore,
            es_client=self.es_client,
            es_index_name="enterprise_knowledge_base"
        )

        # 初始化高级查询拆解器
        self.decomposer = QueryDecomposer(self.llm)
        self.multi_retriever = MultiStepRetriever(
            retriever=self.retriever,
            decomposer=self.decomposer,
            llm=self.llm
        )

        # 初始化重排序器
        self.reranker = EnterpriseReranker() if config.RERANK_ENABLED else None

        logger.info("✅ RAG 核心引擎装配完毕！")

    def ingest_documents(self, request: DocumentIngestRequest) -> Dict[str, Any]:
        """处理并索引文档（双写到 Chroma 和 Elasticsearch）"""
        start_time = time.time()
        logger.info(f"📥 开始入库任务，包含 {len(request.file_paths)} 个文件")

        # 1. 解析与智能分块
        chunks = self.processor.process_batch(request.file_paths)

        # 注入全局元数据
        if request.metadata:
            for chunk in chunks:
                chunk.metadata.update(request.metadata)

        if not chunks:
            logger.warning("⚠️ 未能解析出任何有效的文档块。")
            return {"status": "failed", "message": "无有效内容入库"}

        chunks = filter_complex_metadata(chunks)

        # 2. 写入 ChromaDB (向量库)
        logger.info(f"💾 正在将 {len(chunks)} 个块写入 ChromaDB...")
        self.vectorstore.add_documents(chunks)

        # 3. 写入 Elasticsearch (关键词倒排库)
        if self.es_client:
            logger.info(f"💾 正在将 {len(chunks)} 个块写入 Elasticsearch...")
            # 确保索引存在
            index_name = "enterprise_knowledge_base"
            if not self.es_client.indices.exists(index=index_name):
                self.es_client.indices.create(index=index_name)

            # 批量写入 ES (为简单起见使用循环，生产环境可用 helpers.bulk)
            for i, chunk in enumerate(chunks):
                doc_body = {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                self.es_client.index(index=index_name, id=str(hash(chunk.page_content)), document=doc_body)
            self.es_client.indices.refresh(index=index_name)

        elapsed = time.time() - start_time
        logger.info(f"🎉 入库完成！耗时: {elapsed:.2f}秒, 共处理 {len(chunks)} 个块。")
        return {
            "status": "success",
            "document_count": len(request.file_paths),
            "chunk_count": len(chunks),
            "time_taken": round(elapsed, 2)
        }

    def query(self, request: QueryRequest) -> QueryResponse:
        """执行端到端的高级智能问答查询"""
        start_time = time.time()
        question = request.question
        top_k = request.top_k or config.TOP_K_FINAL

        # 1. 检查 Redis 缓存
        cached_response = self.cache.get(question, request.user_id)
        if cached_response:
            return QueryResponse(**cached_response)

        # 2. 核心召回阶段 (包含直接检索与复杂查询拆解判断)
        if config.QUERY_DECOMPOSITION_ENABLED:
            retrieve_result = self.multi_retriever.retrieve_with_decomposition(
                question=question,
                top_k=config.TOP_K_RETRIEVAL,  # 先召回较多候选集，留给精排
                metadata_filter=request.metadata_filter
            )
            raw_docs = retrieve_result["results"]  # List[Tuple[Document, float]]
            method = retrieve_result["method"]
            answer = retrieve_result.get("aggregated_answer")
        else:
            # 如果关闭了拆解，直接走基础混合检索
            raw_docs = self.retriever.hybrid_search(
                question=question,
                top_k=config.TOP_K_RETRIEVAL,
                metadata_filter=request.metadata_filter
            )
            method = "direct"
            answer = None  # 稍后生成

        # 3. 重排序阶段 (Rerank) 精排
        if self.reranker and config.RERANK_ENABLED and raw_docs:
            logger.info(f"⚖️ 触发交叉编码器重排序 (候选集大小: {len(raw_docs)})...")
            final_docs = self.reranker.rerank(question, raw_docs, top_k=top_k)
        else:
            final_docs = raw_docs[:top_k]

        # 4. 生成最终答案 (如果没走拆解逻辑)
        if not answer:
            if final_docs:
                context = "\n".join([doc.page_content for doc, _ in final_docs])
                prompt = f"基于以下参考信息，请专业、清晰地回答用户问题。\n\n参考信息：\n{context}\n\n用户问题：{question}"
                logger.info("🧠 正在请求大模型生成最终答案...")
                answer = self.llm.invoke(prompt).content
            else:
                answer = "很抱歉，在目前的知识库中未能找到与您问题相关的有效信息。"

        # 5. 组装溯源格式 (Sources)
        sources = []
        confidence = 0.0
        if final_docs:
            confidence = float(final_docs[0][1])  # 最高分
            for i, (doc, score) in enumerate(final_docs):
                sources.append({
                    "id": i + 1,
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "score": round(score, 4),
                    "metadata": doc.metadata
                })

        # 6. 构建并缓存响应对象
        processing_time = time.time() - start_time
        response = QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            processing_time=round(processing_time, 2),
            method=method
        )

        self.cache.set(question, response.model_dump(), request.user_id)
        logger.info(f"✅ 查询处理完毕！总耗时: {processing_time:.2f}秒, 模式: {method}")
        return response