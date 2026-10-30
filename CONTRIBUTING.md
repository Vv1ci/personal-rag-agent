# 贡献与提交

- 提交前运行 `python -m unittest discover -s tests -v`。
- API 使用单 worker，索引更新和检索共享锁。
- 使用清晰提交说明，例如 `feat: add incremental document updates`。
- 通过 `git diff --cached --stat` 检查文件；不提交环境文件、个人资料、模型、对话、数据库和日志。
- `.gitignore` 不会移除已经跟踪的内容或历史提交中的内容。包含个人资料的历史不能直接上传。
- 截图只使用人工测试资料。示例文件须确认有权分享。
- PR 描述写明行为、测试和局限，不把本地测试写成远程 CI 或完整模型问答验证。

- 正式测试放在 `tests/`，人工验收数据放在 `examples/`。学习笔记和 day 阶段脚本保留在本地原项目，不提交到展示仓库。
