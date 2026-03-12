from huggingface_hub import HfApi

def test_hf_token(token):
    api = HfApi()
    try:
        # whoami() 方法会返回该 token 关联的账户信息
        user_info = api.whoami(token=token)
        print("✅ Token 测试成功！验证通过。")
        print(f"👉 关联用户名: {user_info.get('name')}")
        print(f"👉 账号类型: {user_info.get('type')}")
    except Exception as e:
        print("❌ Token 测试失败！")
        print("请检查 Token 是否拼写错误、是否已被吊销，或者网络是否连通。")
        print(f"详细错误信息: {e}")

if __name__ == "__main__":
    # 将下方字符串替换为你的真实 Token (以 hf_ 开头)
    YOUR_TOKEN = "[REDACTED_HF_TOKEN]"
    test_hf_token(YOUR_TOKEN)