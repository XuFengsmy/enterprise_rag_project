# 使用官方 Python 轻量级镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统级依赖 (用于 unstructured 解析 PDF、Word 等复杂文档)
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic-dev \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单并安装
# 注意：确保你的 requirements.txt 中包含了 fastapi, uvicorn, langchain, elasticsearch, redis 等所有依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
# 复制项目所有代码到容器内
COPY . .

# 暴露 FastAPI 默认运行端口
EXPOSE 8000

# 启动命令 (对照你的 main.py 配置)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]