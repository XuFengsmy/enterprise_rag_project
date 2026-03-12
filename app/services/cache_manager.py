import json
import hashlib
import logging
from typing import Optional, Dict
import redis

# 设置基本日志记录
logger = logging.getLogger(__name__)


class CacheManager:
    """Redis 缓存管理器：用于缓存 RAG 系统的查询结果，降低 LLM 成本与响应延迟"""

    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        try:
            # decode_responses=True 确保返回的是字符串而不是字节
            self.redis = redis.from_url(redis_url, decode_responses=True)
            # 测试连接
            self.redis.ping()
            logger.info("✅ Redis 缓存服务连接成功！")
        except Exception as e:
            logger.warning(f"⚠️ Redis 连接失败，缓存功能将被禁用。错误信息: {e}")
            self.redis = None

    def _generate_key(self, question: str, user_id: Optional[str] = None) -> str:
        """生成唯一的缓存键：基于问题内容的 MD5 哈希和可选的用户 ID"""
        key_base = f"rag:{hashlib.md5(question.encode('utf-8')).hexdigest()}"
        if user_id:
            key_base += f":{user_id}"
        return key_base

    def get(self, question: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """获取缓存结果"""
        if not self.redis:
            return None

        try:
            key = self._generate_key(question, user_id)
            data = self.redis.get(key)
            if data:
                logger.info(f"🔍 命中缓存: {key}")
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ 读取 Redis 缓存时发生错误: {e}")
            return None

    def set(self, question: str, result: Dict, user_id: Optional[str] = None):
        """设置缓存结果，包含过期时间"""
        if not self.redis:
            return

        try:
            key = self._generate_key(question, user_id)
            self.redis.setex(key, self.default_ttl, json.dumps(result, ensure_ascii=False))
            logger.info(f"💾 写入缓存: {key}, TTL: {self.default_ttl}s")
        except Exception as e:
            logger.error(f"❌ 写入 Redis 缓存时发生错误: {e}")

    def invalidate(self, question: str, user_id: Optional[str] = None):
        """清除指定问题的缓存"""
        if not self.redis:
            return

        try:
            key = self._generate_key(question, user_id)
            self.redis.delete(key)
            logger.info(f"🗑️ 清除缓存: {key}")
        except Exception as e:
            logger.error(f"❌ 清除 Redis 缓存时发生错误: {e}")