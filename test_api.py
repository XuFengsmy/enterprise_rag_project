import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000"


# 我们把 test_ingest 暂时注释掉，因为你的数据库里已经有数据了！
# def test_ingest(): ...

def test_query():
    print("=" * 50)
    print("🔍 阶段二：测试知识问答 (/v1/query)")
    print("=" * 50)

    payload = {
        "question": "2026年Brand finance全球软实力指数TOP10国家有哪些？",
        "top_k": 3,
        # ⚠️ 关键：换一个全新的 user_id，防止 Redis 返回之前缓存的“没查到”的旧答案
        "user_id": "tester_final_victory"
    }

    print(f"🗣️ 用户提问: {payload['question']}")
    print("🤖 思考中 (正在进行检索、拆解、重排与生成)...\n")

    start_time = time.time()
    response = requests.post(f"{BASE_URL}/v1/query", json=payload)

    if response.status_code == 200:
        data = response.json()
        print("✨ 【AI 回答】:")
        print(f"{data.get('answer')}\n")

        print("-" * 30)
        print(f"⏱️ 总耗时: {data.get('processing_time')} 秒")
        print(f"🎯 检索策略: {data.get('method')} (direct=直接检索, decomposed=复杂拆解)")
        print(f"📈 最高置信度: {data.get('confidence')}")
        print("📚 参考溯源片段:")

        for idx, source in enumerate(data.get('sources', [])):
            content_preview = source['content'].replace('\n', ' ')[:100]
            print(f"   [{idx + 1}] (得分: {source['score']}) {content_preview}...")
    else:
        print(f"❌ 查询失败: {response.text}")


if __name__ == "__main__":
    # 只运行提问测试！
    test_query()