# Personal RAG Agent · 本地个人知识库

基于 FastAPI、Streamlit、Sentence Transformers、ChromaDB 和 Ollama 的个人文档问答项目，支持计算器、文档统计和短期会话记忆。

## 已实现

- TXT、Markdown、文本型 PDF 上传，单文件最大 20 MB。
- 同名替换、相同内容跳过、增量嵌入、删除同步清理分块与向量。
- 更新前完成解析和嵌入；提交异常时尝试恢复原文件与索引。
- 显示来源文件、页码、检索片段与原文件下载。
- Windows 启动中心统一检查环境、初始化模型、启动/停止服务和打开页面。

## 环境与启动

需要 Python 3.11+ 和 [Ollama](https://ollama.com/)。CI 配置为 Python 3.12；桌面环境在 Python 3.14 验证。首次安装依赖和模型需要联网。

1. 首次运行 `安装依赖.cmd`，已有完整环境可直接启动。
2. 可复制 `.env.example` 为 `.env` 修改设置；未创建时使用默认配置。
3. 双击 `启动知识库.cmd`。缺少模型时点击“初始化模型”，完成后自动继续启动。
4. 页面默认在 `http://127.0.0.1:8501`，健康检查为 `http://127.0.0.1:8000/health`。
5. 保留启动中心窗口；停止按钮或关闭窗口会停止本次创建的服务。只关网页不会停止服务。

后续只需双击启动文件，不必重复安装、初始化模型或重建索引。启动器优先使用项目 `.venv`；否则使用已安装的 Python。原已运行的 Ollama 不会被启动器关闭。

## 手动启动

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5:1.5b
.venv\Scripts\python.exe download_embedding_model.py
```

确认 Ollama 服务运行后，在两个终端分别运行：

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
.venv\Scripts\python.exe -m streamlit run ui/streamlit_app.py --server.address 127.0.0.1
```

API 必须使用 **一个 worker**，以便更新和检索共享进程锁。

## 配置与结构

| 变量 | 默认值 |
| --- | --- |
| OLLAMA_BASE_URL | http://127.0.0.1:11434 |
| OLLAMA_MODEL | qwen2.5:1.5b |
| EMBEDDING_MODEL | sentence-transformers/all-MiniLM-L6-v2 |
| DEFAULT_TOP_K | 3 |
| MIN_RELEVANCE_SCORE | 0.42 |
| DATA_RAW_DIR | data/raw |
| DATA_PROCESSED_DIR | data/processed |
| CHROMA_DB_DIR | chroma_db |

更多变量见 `.env.example`。切换嵌入模型时另建向量目录并重新索引，避免新旧向量不兼容。

```text
launcher.py / scripts/launch.ps1  Windows 启动管理
api/                            HTTP 接口
ui/                             Streamlit 页面
rag/agent/                      工具路由与记忆
rag/documents.py                增量文档更新、删除与回滚
rag/ingestion.py                解析与分块
rag/embeddings/                 嵌入模型源代码
rag/retrieval/                  向量检索
rag/chains/                     上下文、来源与生成
tests/                          单元与接口测试
examples/                       人工测试文档
```

上传 → 解析分块 → 嵌入 → ChromaDB；问题 → 检索 → Ollama → 答案与来源。个人文件及派生内容不随仓库分发。

## 验证

```powershell
python -m unittest discover -s tests -v
```

测试覆盖重复上传、替换、删除、路径校验、失败恢复，以及真实 Chroma 与 FastAPI 接口生命周期。测试使用临时库和固定嵌入，不调用 Ollama、不下载模型，也不改个人知识库。独立测试依赖见 `requirements-test.txt`。

GitHub Actions 自动运行语法检查和独立测试。仓库保留 `tests/` 中可独立运行的正式测试；开发阶段的 day 脚本、教学索引示例、临时审计及快照工具不随正式源码分发。

可选命令行工具为 `ask_rag.py`（提问）、`run_api.py`（启动 API）、`validate_chunks.py`（检查分块结构）；日常文档更新使用页面/API 的增量索引流程。

人工验收见 [使用说明](docs/USAGE.md)，可使用 `examples/acceptance.txt`。

## 限制

- 面向本机单用户，未实现公网账户权限体系，请保持本机绑定。
- 文件系统与向量库不提供断电事务保证；原文档应自行备份，异常恢复后可重新索引。
- 扫描 PDF 需要先 OCR；目前没有内置 OCR。
- 默认小型嵌入模型在中文等语言上的检索质量需要实际评估。检索分数不代表答案正确概率。
- 初始化会访问依赖/模型下载服务；配置远程模型服务后，相关内容会发送到该地址。
- 仓库不包含模型权重、个人文件、分块、对话、运行库及私人截图。

提交规范见 [贡献指南](CONTRIBUTING.md)。第三方依赖与模型适用各自许可证；当前未另行授予项目级开源许可。
