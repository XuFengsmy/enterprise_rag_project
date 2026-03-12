import logging
from typing import List, Tuple

from langchain_core.documents import Document

# 设置日志
logger = logging.getLogger(__name__)


class EnterpriseReranker:
    """企业级重排序器：使用 Cross-Encoder 对初步召回结果进行二次精准打分"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str = "cpu"):
        """
        初始化重排序模型
        :param model_name: HuggingFace 或本地模型路径
        :param device: 运行设备 ('cpu', 'cuda', 'mps' 等)
        """
        logger.info(f"⏳ 正在初始化重排序模型: {model_name} (计算设备: {device})")
        try:
            # 采用局部导入：如果在配置中禁用了 Rerank，就不会拖慢整个应用的启动速度
            from sentence_transformers import CrossEncoder

            # 初始化模型（默认不限制最大长度，实际生产可根据显存调整 max_length）
            self.model = CrossEncoder(model_name, device=device)
            logger.info("✅ 重排序模型加载完成！")
        except ImportError:
            logger.error("❌ 未安装 sentence_transformers，请执行: pip install sentence-transformers")
            self.model = None
        except Exception as e:
            logger.error(f"❌ 重排序模型加载失败: {e}")
            self.model = None

    def rerank(self, query: str, documents: List[Tuple[Document, float]], top_k: int = 3) -> List[
        Tuple[Document, float]]:
        """
        对混合检索的初步结果进行重排序

        :param query: 用户原始查询
        :param documents: 混合检索返回的 [(Document, 融合分数), ...] 列表
        :param top_k: 最终保留的文档数量
        :return: 重排序后的 [(Document, 精排分数), ...] 列表
        """
        if not documents:
            logger.debug("传入的文档列表为空，跳过重排序。")
            return []

        if not self.model:
            logger.warning("⚠️ 重排序模型未就绪，安全降级：直接返回粗排 Top-K 结果。")
            return documents[:top_k]

        try:
            # 1. 构建 Cross-Encoder 需要的文本对: [[query, doc_text1], [query, doc_text2], ...]
            pairs = [[query, doc.page_content] for doc, _ in documents]

            # 2. 批量预测相关性分数 (精排分数)
            logger.info(f"⚖️ 开始对 {len(pairs)} 条候选文档进行交叉编码精排...")
            scores = self.model.predict(pairs)

            # 3. 将新生成的分数与原始 Document 对象重新绑定
            reranked_results = []
            for i, score in enumerate(scores):
                # 保持 (Document, float) 的数据结构统一
                reranked_results.append((documents[i][0], float(score)))

            # 4. 根据精排分数降序排列
            reranked_results.sort(key=lambda x: x[1], reverse=True)

            logger.info(f"✨ 重排序完成！已截取前 {top_k} 条最高质量的结果。")
            return reranked_results[:top_k]

        except Exception as e:
            logger.error(f"❌ 重排序过程中发生异常: {e}")
            # 发生异常时进行安全降级，防止整个查询接口崩溃
            return documents[:top_k]