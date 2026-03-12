import json
import logging
from typing import Dict, List, Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel

# 设置日志
logger = logging.getLogger(__name__)


class QueryDecomposer:
    """查询分解专家：负责将复杂问题拆解，并最终综合所有子答案"""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def decompose(self, question: str, num_subqueries: int = 3) -> List[str]:
        """将复杂问题拆解为多个子问题"""
        prompt = ChatPromptTemplate.from_template("""
        你是一个知识库查询分解专家。请将以下复杂问题分解为不超过 {num} 个更简单、更聚焦的子问题。
        每个子问题应该能够独立检索和回答，并且组合起来可以回答原问题。

        原问题：{question}

        请严格以 JSON 数组格式输出子问题列表，例如 ["子问题1", "子问题2"]。
        只输出 JSON，不要包含任何 markdown 标记或其他解释性文字。
        """)
        chain = prompt | self.llm

        logger.info(f"🧠 正在分析并拆解复杂问题: '{question}'")
        try:
            response = chain.invoke({"question": question, "num": num_subqueries})
            # 清理可能存在的 markdown 标记 (如 ```json ... ```)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()

            subqueries = json.loads(content)

            if isinstance(subqueries, list) and len(subqueries) > 0:
                logger.info(f"✅ 成功拆解为 {len(subqueries)} 个子问题: {subqueries}")
                return subqueries
            return [question]
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ 大模型输出 JSON 解析失败，降级为原问题: {e}")
            return [question]
        except Exception as e:
            logger.error(f"❌ 拆解问题时发生未知错误: {e}")
            return [question]

    def aggregate_answers(self, question: str, sub_answers: List[Dict]) -> str:
        """基于所有子问题的临时答案，综合生成最终回答"""
        context = "\n\n".join([
            f"【子问题 {i + 1}】: {item['question']}\n【参考答案】: {item['answer']}"
            for i, item in enumerate(sub_answers)
        ])

        prompt = ChatPromptTemplate.from_template("""
        基于以下子问题和对应的参考答案，请综合回答用户的原始问题。
        请确保最终答案全面、连贯，并且不要在回答中暴露你拆解了问题这件事情。

        原问题：{question}

        相关参考信息：
        {context}

        请给出最终的详细回答：
        """)
        chain = prompt | self.llm

        logger.info(f"🧩 正在聚合 {len(sub_answers)} 个子答案以生成最终回复...")
        response = chain.invoke({"question": question, "context": context})
        return response.content


class MultiStepRetriever:
    """多步智能检索器：融合了基础检索与查询分解的调度枢纽"""

    def __init__(self, retriever, decomposer: QueryDecomposer, llm: BaseChatModel, direct_threshold: float = 0.03):
        self.retriever = retriever
        self.decomposer = decomposer
        self.llm = llm
        # RRF 分数通常较小，0.03 是一个经验阈值，高于此说明基础检索已经非常确信了
        self.direct_threshold = direct_threshold

    def retrieve_with_decomposition(self, question: str, top_k: int = 3,
                                    metadata_filter: Optional[Dict[str, Any]] = None) -> Dict:
        """执行带智能判断的多步检索"""

        # 步骤 1：先尝试一次直接混合检索
        logger.info("🚀 执行首次直接混合检索探测...")
        direct_results = self.retriever.hybrid_search(question, top_k, metadata_filter)

        # 步骤 2：判断是否足够自信（分数是否大于阈值）
        if direct_results and len(direct_results) >= top_k and direct_results[0][1] > self.direct_threshold:
            logger.info(
                f"⚡ 首选命中得分 ({direct_results[0][1]:.4f}) 高于阈值 ({self.direct_threshold})，直接返回基础检索结果。")
            return {
                "method": "direct",
                "results": direct_results,
                "subqueries": []
            }

        # 步骤 3：如果不够自信，触发复杂拆解流程
        logger.info(f"🐢 首选命中得分不足，触发复杂问题多步拆解流程...")
        subqueries = self.decomposer.decompose(question)

        all_results = []
        sub_answers = []

        # 步骤 4：对每个子问题分别检索并生成中间答案
        for sq in subqueries:
            sq_results = self.retriever.hybrid_search(sq, top_k, metadata_filter)
            all_results.extend(sq_results)

            # 提取前两条片段喂给 LLM 做临时总结
            context = "\n".join([doc.page_content for doc, _ in sq_results[:2]])
            if context.strip():
                answer = self.llm.invoke(f"基于以下信息回答问题：{sq}\n信息：{context}").content
            else:
                answer = "未找到相关信息。"

            sub_answers.append({
                "question": sq,
                "answer": answer,
                "results": sq_results
            })

        # 步骤 5：大模型聚合最终答案
        final_answer = self.decomposer.aggregate_answers(question, sub_answers)

        # 步骤 6：对所有参考过的文档进行去重和排序 (保持 Document 对象结构)
        seen_content = set()
        unique_results = []
        # 按分数从高到低排序
        for doc, score in sorted(all_results, key=lambda x: x[1], reverse=True):
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append((doc, score))

        logger.info("🎉 多步拆解与聚合流程处理完毕！")
        return {
            "method": "decomposed",
            "results": unique_results[:top_k],  # 只返回最终的 Top K 溯源文档
            "subqueries": subqueries,
            "sub_answers": sub_answers,
            "aggregated_answer": final_answer
        }