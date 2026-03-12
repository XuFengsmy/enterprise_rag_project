import os
import logging
from typing import List, Dict, Any
from datetime import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_unstructured import UnstructuredLoader

# 设置日志
logger = logging.getLogger(__name__)


class EnterpriseDocumentProcessor:
    """企业级文档处理器：支持多种格式、智能分块、元数据提取"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 基础分块器：适用于长文本和不支持结构化解析的文档
        self.base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )

        # Markdown分块器：保留标题层级结构（特别适合处理规范的技术文档）
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3"),
            ]
        )
        logger.info(f"📄 初始化文档处理器 (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")

    def load_document(self, file_path: str) -> List[Document]:
        """加载文档，利用 Unstructured 自动识别格式 (PDF, Word, MD等)"""
        try:
            logger.info(f"⏳ 正在加载文档: {file_path}")
            # mode="elements" 可以将文档解析为具有层级关系的元素
            loader = UnstructuredLoader(file_path, mode="elements")
            documents = loader.load()
            logger.info(f"✅ 成功加载文档: {file_path}, 共解析出 {len(documents)} 个底层元素")
            return documents
        except Exception as e:
            logger.error(f"❌ 加载文档失败 {file_path}: {e}")
            raise e

    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """提取并标准化文档元数据"""
        metadata = document.metadata.copy()

        # 添加系统级处理时间
        metadata["processed_at"] = datetime.now().isoformat()

        # 提取文档扩展名类型
        source = metadata.get("source", "")
        if source:
            file_ext = source.split(".")[-1].lower()
            metadata["file_type"] = file_ext

        # 提取 Markdown 章节信息（如果有）
        if "header1" in metadata:
            metadata["chapter"] = metadata["header1"]

        return metadata

    def smart_split(self, documents: List[Document]) -> List[Document]:
        """智能分块：根据文档类型和内容特征选择最优分块策略"""
        all_chunks = []

        for doc in documents:
            # 兼容 Unstructured 返回的 Element 和常规的 Document
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            metadata = self.extract_metadata(doc)

            # 策略1：如果是 Markdown 格式且包含标题，使用标题结构化分块
            if content.startswith("#") or "\n#" in content:
                try:
                    md_chunks = self.md_splitter.split_text(content)
                    for chunk in md_chunks:
                        chunk.metadata.update(metadata)
                        all_chunks.append(chunk)
                    continue  # 处理成功则跳过后续策略
                except Exception as e:
                    logger.debug(f"Markdown分块失败，降级为基础分块: {e}")

            # 策略2：超长文本片段使用基础滑动窗口分块
            if len(content) > self.chunk_size:
                chunks = self.base_splitter.create_documents([content], [metadata])
                all_chunks.extend(chunks)

            # 策略3：短文档或短元素直接保留
            else:
                # 重新构造标准 LangChain Document 以确保类型一致
                new_doc = Document(page_content=content, metadata=metadata)
                all_chunks.append(new_doc)

        logger.info(f"✂️ 智能分块完成: 原始 {len(documents)} 个元素 -> 最终 {len(all_chunks)} 个切块 (Chunks)")
        return all_chunks

    def process_batch(self, file_paths: List[str]) -> List[Document]:
        """批量处理多个本地文档（常用于入库接口）"""
        all_chunks = []
        for path in file_paths:
            if not os.path.exists(path):
                logger.warning(f"⚠️ 文件不存在，跳过: {path}")
                continue

            docs = self.load_document(path)
            chunks = self.smart_split(docs)
            all_chunks.extend(chunks)

        return all_chunks