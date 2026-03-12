> **🤖 自动化声明**：本仓库的代码测试、环境整理及自动推送，均由我本地部署的 OpenClaw 智能体全自动协同完成。

# 企业级 RAG 项目 (Enterprise RAG Project)

这是一个用于构建企业级检索增强生成（RAG）系统的完整项目模板。它包含了从数据预处理、向量存储、混合检索、重排序到 API 服务的全套模块，并支持 Docker 部署。

## 📁 项目结构

```
enterprise_rag_project/
├── app/                    # FastAPI 应用核心
│   ├── main.py             # 应用入口
│   ├── core/               # 核心配置
│   │   └── config.py
│   ├── models/             # 数据模型
│   │   └── schemas.py
│   ├── api/                # API 路由
│   │   └── routes.py
│   └── rag/                # RAG 核心逻辑
│       ├── engine.py
│       ├── document_processor.py
│       ├── hybrid_retriever.py
│       ├── query_decomposer.py
│       └── reranker.py
├── huggingface-test/       # Hugging Face Token 测试脚本
│   └── test1.py
├── memory/                 # 运行时记忆文件
│   └── 2026-03-12.md
├── tests/                  # 单元测试 (预留)
├── Dockerfile              # Docker 镜像构建文件
├── docker-compose.yml      # Docker 服务编排
├── requirements.txt        # Python 依赖
├── README.md               # 项目说明 (你正在阅读的文件)
└── .gitignore              # Git 忽略规则
```

## 🚀 快速开始

1. **安装依赖**:
   ```bash
   pip install -r requirements.txt
   ```

2. **启动服务**:
   ```bash
   python app/main.py
   ```

3. **访问 API**:
   服务默认运行在 `http://localhost:8000`。

## 🧪 测试

项目包含基础的单元测试框架，位于 `tests/` 目录下。

## 🐳 Docker 部署

项目支持通过 Docker 进行容器化部署，简化环境配置。

```bash
# 构建镜像
docker build -t enterprise-rag .

# 启动服务
docker-compose up
```

## 🤖 关于自动化

本项目的初始化、代码整理和首次推送，由 OpenClaw 智能体辅助完成，旨在展示 AI 在软件工程流程中的实际应用价值。