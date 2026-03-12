import pytest
from unittest.mock import Mock, patch
from app.rag.hybrid_retriever import EnterpriseHybridRetriever
from langchain_core.documents import Document

# 测试EnterpriseHybridRetriever类
class TestEnterpriseHybridRetriever:
    
    def test_init_with_valid_es_client(self):
        """测试初始化时Elasticsearch连接正常"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.es_client == mock_es_client
        assert retriever.rrf_k == 60
    
    def test_init_with_invalid_es_client(self):
        """测试初始化时Elasticsearch连接失败"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = False
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.es_client is None
    
    def test_init_with_es_exception(self):
        """测试初始化时Elasticsearch抛出异常"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.side_effect = Exception("Connection error")
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        
        assert retriever.vectorstore == mock_vectorstore
        assert retriever.es_client is None
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_vector_search_success(self, mock_logger):
        """测试向量检索成功"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        # 创建模拟文档
        mock_doc1 = Document(page_content="test content 1", metadata={})
        mock_doc2 = Document(page_content="test content 2", metadata={})
        
        # 设置相似度搜索返回值
        mock_vectorstore.similarity_search.return_value = [mock_doc1, mock_doc2]
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        results = retriever._vector_search("test query", 5)
        
        assert len(results) == 2
        assert results[0].page_content == "test content 1"
        assert results[1].page_content == "test content 2"
        mock_vectorstore.similarity_search.assert_called_once()
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_vector_search_with_filter(self, mock_logger):
        """测试带过滤条件的向量检索"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        # 创建模拟文档
        mock_doc = Document(page_content="filtered content", metadata={})
        mock_vectorstore.similarity_search.return_value = [mock_doc]
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        filter_dict = {"department": "IT"}
        results = retriever._vector_search("test query", 5, filter_dict)
        
        assert len(results) == 1
        mock_vectorstore.similarity_search.assert_called_once_with(
            "test query", k=5, filter=filter_dict
        )
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_vector_search_failure(self, mock_logger):
        """测试向量检索失败"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        # 设置相似度搜索抛出异常
        mock_vectorstore.similarity_search.side_effect = Exception("Search failed")
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        results = retriever._vector_search("test query", 5)
        
        assert len(results) == 0
        mock_logger.error.assert_called_once()
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_keyword_search_success(self, mock_logger):
        """测试关键词检索成功"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        mock_es_client.indices.exists.return_value = True
        
        # 设置ES搜索返回值
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "content": "keyword content 1",
                            "metadata": {}
                        }
                    },
                    {
                        "_source": {
                            "content": "keyword content 2",
                            "metadata": {}
                        }
                    }
                ]
            }
        }
        mock_es_client.search.return_value = mock_response
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        results = retriever._keyword_search("test query", 5)
        
        assert len(results) == 2
        assert results[0].page_content == "keyword content 1"
        assert results[1].page_content == "keyword content 2"
        mock_es_client.search.assert_called_once()
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_keyword_search_with_filter(self, mock_logger):
        """测试带过滤条件的关键词检索"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        mock_es_client.indices.exists.return_value = True
        
        # 设置ES搜索返回值
        mock_response = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "content": "filtered keyword content",
                            "metadata": {}
                        }
                    }
                ]
            }
        }
        mock_es_client.search.return_value = mock_response
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        filter_dict = {"department": "IT"}
        results = retriever._keyword_search("test query", 5, filter_dict)
        
        assert len(results) == 1
        mock_es_client.search.assert_called_once()
        # 验证查询中包含过滤条件
        call_args = mock_es_client.search.call_args[1]
        assert "filter" in call_args["query"]["bool"]
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_keyword_search_no_es_client(self, mock_logger):
        """测试没有Elasticsearch客户端时的关键词检索"""
        mock_vectorstore = Mock()
        retriever = EnterpriseHybridRetriever(mock_vectorstore, None)
        
        results = retriever._keyword_search("test query", 5)
        
        assert len(results) == 0
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_keyword_search_index_not_exists(self, mock_logger):
        """测试索引不存在时的关键词检索"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        mock_es_client.indices.exists.return_value = False
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        results = retriever._keyword_search("test query", 5)
        
        assert len(results) == 0
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_keyword_search_failure(self, mock_logger):
        """测试关键词检索失败"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        mock_es_client.indices.exists.return_value = True
        
        # 设置ES搜索抛出异常
        mock_es_client.search.side_effect = Exception("Search failed")
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        results = retriever._keyword_search("test query", 5)
        
        assert len(results) == 0
        mock_logger.error.assert_called_once()
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_hybrid_search_basic(self, mock_logger):
        """测试基本的混合检索功能"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        mock_es_client.indices.exists.return_value = True
        
        # 设置向量检索返回值
        mock_vector_doc = Document(page_content="vector content", metadata={})
        mock_vectorstore.similarity_search.return_value = [mock_vector_doc]
        
        # 设置关键词检索返回值
        mock_keyword_doc = Document(page_content="keyword content", metadata={})
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "content": "keyword content",
                            "metadata": {}
                        }
                    }
                ]
            }
        }
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        results = retriever.hybrid_search("test query", 2)
        
        assert len(results) == 2
        # 验证结果包含两种类型的文档
        contents = [doc.page_content for doc, score in results]
        assert "vector content" in contents
        assert "keyword content" in contents
        
        # 验证调用了两次检索方法
        assert mock_vectorstore.similarity_search.call_count == 1
        assert mock_es_client.search.call_count == 1
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_hybrid_search_with_empty_filter(self, mock_logger):
        """测试带有空过滤条件的混合检索"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        mock_es_client.indices.exists.return_value = True
        
        # 设置检索返回值
        mock_vector_doc = Document(page_content="vector content", metadata={})
        mock_vectorstore.similarity_search.return_value = [mock_vector_doc]
        
        mock_keyword_doc = Document(page_content="keyword content", metadata={})
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "content": "keyword content",
                            "metadata": {}
                        }
                    }
                ]
            }
        }
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        # 传递空字典作为过滤条件
        results = retriever.hybrid_search("test query", 2, {})
        
        assert len(results) == 2
        # 验证过滤条件被正确处理（应该被置为None）
        mock_vectorstore.similarity_search.assert_called_once_with("test query", k=4, filter=None)
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_hybrid_search_with_metadata_filter(self, mock_logger):
        """测试带有元数据过滤条件的混合检索"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        mock_es_client.indices.exists.return_value = True
        
        # 设置检索返回值
        mock_vector_doc = Document(page_content="vector content", metadata={})
        mock_vectorstore.similarity_search.return_value = [mock_vector_doc]
        
        mock_keyword_doc = Document(page_content="keyword content", metadata={})
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "content": "keyword content",
                            "metadata": {}
                        }
                    }
                ]
            }
        }
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        # 传递有效的过滤条件
        filter_dict = {"department": "IT"}
        results = retriever.hybrid_search("test query", 2, filter_dict)
        
        assert len(results) == 2
        # 验证过滤条件被正确传递
        mock_vectorstore.similarity_search.assert_called_once_with("test query", k=4, filter=filter_dict)
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_hybrid_search_with_dirty_filter(self, mock_logger):
        """测试带有脏数据过滤条件的混合检索"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        mock_es_client.indices.exists.return_value = True
        
        # 设置检索返回值
        mock_vector_doc = Document(page_content="vector content", metadata={})
        mock_vectorstore.similarity_search.return_value = [mock_vector_doc]
        
        mock_keyword_doc = Document(page_content="keyword content", metadata={})
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "content": "keyword content",
                            "metadata": {}
                        }
                    }
                ]
            }
        }
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        # 传递Swagger UI传来的脏数据
        dirty_filter = {"additionalProp1": {}}
        results = retriever.hybrid_search("test query", 2, dirty_filter)
        
        assert len(results) == 2
        # 验证脏数据被正确清洗（应该被置为None）
        mock_vectorstore.similarity_search.assert_called_once_with("test query", k=4, filter=None)
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_hybrid_search_no_results(self, mock_logger):
        """测试没有检索结果的情况"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        
        # 设置检索返回空结果
        mock_vectorstore.similarity_search.return_value = []
        mock_es_client.search.return_value = {"hits": {"hits": []}}
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        results = retriever.hybrid_search("test query", 2)
        
        assert len(results) == 0
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_hybrid_search_only_vector_results(self, mock_logger):
        """测试只有向量检索有结果的情况"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        mock_es_client.indices.exists.return_value = True
        
        # 设置向量检索有结果，关键词检索无结果
        mock_vector_doc = Document(page_content="vector content", metadata={})
        mock_vectorstore.similarity_search.return_value = [mock_vector_doc]
        mock_es_client.search.return_value = {"hits": {"hits": []}}
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        results = retriever.hybrid_search("test query", 2)
        
        assert len(results) == 1
        assert results[0][0].page_content == "vector content"
    
    @patch('app.rag.hybrid_retriever.logger')
    def test_hybrid_search_only_keyword_results(self, mock_logger):
        """测试只有关键词检索有结果的情况"""
        mock_vectorstore = Mock()
        mock_es_client = Mock()
        mock_es_client.ping.return_value = True
        mock_es_client.indices.exists.return_value = True
        
        # 设置关键词检索有结果，向量检索无结果
        mock_vectorstore.similarity_search.return_value = []
        
        mock_keyword_doc = Document(page_content="keyword content", metadata={})
        mock_es_client.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "content": "keyword content",
                            "metadata": {}
                        }
                    }
                ]
            }
        }
        
        retriever = EnterpriseHybridRetriever(mock_vectorstore, mock_es_client)
        results = retriever.hybrid_search("test query", 2)
        
        assert len(results) == 1
        assert results[0][0].page_content == "keyword content"