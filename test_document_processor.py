import pytest
from unittest.mock import Mock, patch
import os
from app.rag.document_processor import EnterpriseDocumentProcessor
from langchain_core.documents import Document

# 测试EnterpriseDocumentProcessor类
class TestEnterpriseDocumentProcessor:
    
    def test_init_default_parameters(self):
        """测试默认参数初始化"""
        processor = EnterpriseDocumentProcessor()
        
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 100
        assert hasattr(processor, 'base_splitter')
        assert hasattr(processor, 'md_splitter')
    
    def test_init_custom_parameters(self):
        """测试自定义参数初始化"""
        processor = EnterpriseDocumentProcessor(chunk_size=1000, chunk_overlap=200)
        
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
    
    @patch('app.rag.document_processor.UnstructuredLoader')
    @patch('app.rag.document_processor.logger')
    def test_load_document_success(self, mock_logger, mock_loader):
        """测试文档加载成功"""
        # 创建模拟文档
        mock_doc = Mock()
        mock_loader.return_value.load.return_value = [mock_doc]
        
        processor = EnterpriseDocumentProcessor()
        file_path = "test.pdf"
        results = processor.load_document(file_path)
        
        assert len(results) == 1
        assert results[0] == mock_doc
        mock_loader.assert_called_once_with(file_path, mode="elements")
        mock_logger.info.assert_called()
    
    @patch('app.rag.document_processor.UnstructuredLoader')
    @patch('app.rag.document_processor.logger')
    def test_load_document_failure(self, mock_logger, mock_loader):
        """测试文档加载失败"""
        # 设置加载器抛出异常
        mock_loader.side_effect = Exception("Load failed")
        
        processor = EnterpriseDocumentProcessor()
        file_path = "nonexistent.pdf"
        
        with pytest.raises(Exception) as exc_info:
            processor.load_document(file_path)
        
        assert str(exc_info.value) == "Load failed"
        mock_logger.error.assert_called()
    
    def test_extract_metadata_basic(self):
        """测试基本元数据提取"""
        processor = EnterpriseDocumentProcessor()
        
        # 创建测试文档
        doc = Document(page_content="test content", metadata={"source": "test.pdf"})
        
        metadata = processor.extract_metadata(doc)
        
        assert "processed_at" in metadata
        assert metadata["file_type"] == "pdf"
        assert metadata["source"] == "test.pdf"
    
    def test_extract_metadata_with_headers(self):
        """测试带标题的元数据提取"""
        processor = EnterpriseDocumentProcessor()
        
        # 创建带标题的测试文档
        doc = Document(
            page_content="test content", 
            metadata={
                "source": "test.md",
                "header1": "Test Header"
            }
        )
        
        metadata = processor.extract_metadata(doc)
        
        assert metadata["chapter"] == "Test Header"
        assert metadata["file_type"] == "md"
    
    def test_extract_metadata_no_source(self):
        """测试没有source字段的元数据提取"""
        processor = EnterpriseDocumentProcessor()
        
        # 创建无source的测试文档
        doc = Document(page_content="test content", metadata={})
        
        metadata = processor.extract_metadata(doc)
        
        assert "processed_at" in metadata
        assert "file_type" not in metadata
    
    def test_smart_split_markdown_with_headers(self):
        """测试Markdown格式带标题的智能分块"""
        processor = EnterpriseDocumentProcessor()
        
        # 创建Markdown格式文档
        content = "# Header 1\nContent under header 1\n## Header 2\nContent under header 2"
        doc = Document(page_content=content, metadata={})
        
        chunks = processor.smart_split([doc])
        
        # 应该被分成多个块，每个块都有对应的标题元数据
        assert len(chunks) >= 2
        # 验证第一个块有header1元数据
        assert "header1" in chunks[0].metadata
        assert chunks[0].metadata["header1"] == "Header 1"
    
    def test_smart_split_long_text(self):
        """测试长文本的智能分块"""
        processor = EnterpriseDocumentProcessor(chunk_size=50, chunk_overlap=10)
        
        # 创建长文本文档
        long_content = "This is a very long text that should be split into multiple chunks. " * 10
        doc = Document(page_content=long_content, metadata={})
        
        chunks = processor.smart_split([doc])
        
        # 长文本应该被分割成多个块
        assert len(chunks) > 1
        # 每个块的长度应该在合理范围内
        for chunk in chunks:
            assert len(chunk.page_content) <= 50 + 10  # chunk_size + some buffer
    
    def test_smart_split_short_text(self):
        """测试短文本的智能分块"""
        processor = EnterpriseDocumentProcessor()
        
        # 创建短文本文档
        short_content = "Short text"
        doc = Document(page_content=short_content, metadata={})
        
        chunks = processor.smart_split([doc])
        
        # 短文本应该保持为一个块
        assert len(chunks) == 1
        assert chunks[0].page_content == short_content
    
    def test_smart_split_non_markdown(self):
        """测试非Markdown格式文本的智能分块"""
        processor = EnterpriseDocumentProcessor()
        
        # 创建非Markdown格式文档
        content = "Plain text without any headers\nJust some content here"
        doc = Document(page_content=content, metadata={})
        
        chunks = processor.smart_split([doc])
        
        # 非Markdown文本应该使用基础分块器
        assert len(chunks) >= 1
        # 验证没有添加额外的标题元数据
        for chunk in chunks:
            assert "header1" not in chunk.metadata
    
    def test_smart_split_empty_content(self):
        """测试空内容的智能分块"""
        processor = EnterpriseDocumentProcessor()
        
        # 创建空内容文档
        doc = Document(page_content="", metadata={})
        
        chunks = processor.smart_split([doc])
        
        # 空内容应该返回一个空块
        assert len(chunks) == 1
        assert chunks[0].page_content == ""
    
    @patch('app.rag.document_processor.logger')
    def test_smart_split_markdown_failure_fallback(self, mock_logger):
        """测试Markdown分块失败时回退到基础分块"""
        processor = EnterpriseDocumentProcessor()
        
        # 创建会导致Markdown分块失败的内容
        content = "# Invalid Markdown\nSome content"
        doc = Document(page_content=content, metadata={})
        
        # 模拟Markdown分块器抛出异常
        with patch.object(processor.md_splitter, 'split_text', side_effect=Exception("Split failed")):
            chunks = processor.smart_split([doc])
        
        # 应该回退到基础分块
        assert len(chunks) >= 1
        mock_logger.debug.assert_called_with("Markdown分块失败，降级为基础分块: Split failed")
    
    @patch('app.rag.document_processor.os.path.exists')
    @patch('app.rag.document_processor.logger')
    def test_process_batch_success(self, mock_logger, mock_exists):
        """测试批量处理文档成功"""
        mock_exists.return_value = True
        
        processor = EnterpriseDocumentProcessor()
        
        # 模拟load_document和smart_split方法
        with patch.object(processor, 'load_document') as mock_load, \
             patch.object(processor, 'smart_split') as mock_split:
            
            # 创建模拟文档
            mock_doc1 = Document(page_content="content 1", metadata={})
            mock_doc2 = Document(page_content="content 2", metadata={})
            
            # 设置返回值
            mock_load.return_value = [mock_doc1]
            mock_split.return_value = [mock_doc2]
            
            file_paths = ["file1.pdf", "file2.docx"]
            results = processor.process_batch(file_paths)
            
            assert len(results) == 2
            assert results[0] == mock_doc2
            assert results[1] == mock_doc2
            
            # 验证每个文件都被处理
            assert mock_load.call_count == 2
            assert mock_split.call_count == 2
    
    @patch('app.rag.document_processor.os.path.exists')
    @patch('app.rag.document_processor.logger')
    def test_process_batch_file_not_exists(self, mock_logger, mock_exists):
        """测试批量处理时文件不存在的情况"""
        mock_exists.return_value = False
        
        processor = EnterpriseDocumentProcessor()
        
        # 模拟load_document和smart_split方法
        with patch.object(processor, 'load_document') as mock_load, \
             patch.object(processor, 'smart_split') as mock_split:
            
            file_paths = ["nonexistent.pdf"]
            results = processor.process_batch(file_paths)
            
            assert len(results) == 0
            mock_load.assert_not_called()
            mock_split.assert_not_called()
            mock_logger.warning.assert_called_with("⚠️ 文件不存在，跳过: nonexistent.pdf")
    
    @patch('app.rag.document_processor.os.path.exists')
    @patch('app.rag.document_processor.logger')
    def test_process_batch_mixed_files(self, mock_logger, mock_exists):
        """测试批量处理混合存在和不存在的文件"""
        def mock_exists_func(path):
            return path != "nonexistent.pdf"
        
        mock_exists.side_effect = mock_exists_func
        
        processor = EnterpriseDocumentProcessor()
        
        # 模拟load_document和smart_split方法
        with patch.object(processor, 'load_document') as mock_load, \
             patch.object(processor, 'smart_split') as mock_split:
            
            # 创建模拟文档
            mock_doc = Document(page_content="content", metadata={})
            
            # 设置返回值
            mock_load.return_value = [mock_doc]
            mock_split.return_value = [mock_doc]
            
            file_paths = ["file1.pdf", "nonexistent.pdf", "file2.docx"]
            results = processor.process_batch(file_paths)
            
            assert len(results) == 2  # 只有存在的文件被处理
            assert mock_load.call_count == 2
            assert mock_split.call_count == 2
            mock_logger.warning.assert_called_with("⚠️ 文件不存在，跳过: nonexistent.pdf")
    
    @patch('app.rag.document_processor.os.path.exists')
    @patch('app.rag.document_processor.logger')
    def test_process_batch_empty_list(self, mock_logger, mock_exists):
        """测试批量处理空文件列表"""
        processor = EnterpriseDocumentProcessor()
        
        results = processor.process_batch([])
        
        assert len(results) == 0
        mock_exists.assert_not_called()
    
    @patch('app.rag.document_processor.os.path.exists')
    @patch('app.rag.document_processor.logger')
    def test_process_batch_all_files_missing(self, mock_logger, mock_exists):
        """测试批量处理所有文件都不存在的情况"""
        mock_exists.return_value = False
        
        processor = EnterpriseDocumentProcessor()
        
        file_paths = ["file1.pdf", "file2.docx", "file3.txt"]
        results = processor.process_batch(file_paths)
        
        assert len(results) == 0
        assert mock_logger.warning.call_count == 3