# MiniRAG

一个可运行的小型中文 RAG 问答项目：文档解析、结构化分块、BM25 + BGE-M3 向量混合检索、RRF 融合、rerank，以及带来源引用的 DeepSeek 回答。

项目包含一个 FastAPI 后端和一个不需要 Node.js 构建的 HTML/CSS/JavaScript 单页前端。

## 功能

- 支持网页上传和本地目录导入 Markdown / TXT 文档
- 结构化 Markdown 分块，保留标题路径和原文位置
- BM25 + SiliconFlow `BAAI/bge-m3` 向量检索
- RRF 融合和 rerank
- DeepSeek 带 `[n]` 引用回答
- 点击回答中的引用编号查看证据原文
- 文档指纹缓存索引，文档变化后自动失效
- 评测题目集独立存储，可上传替换，不把题目硬编码在 Python 中
- 评测支持“可答”和“拒答”两类题目

## 项目结构

```text
app/
  main.py              FastAPI 路由
  kb.py                知识库、索引、问答、评测服务
  static/              单页前端
chunking/              Markdown 结构化分块
docparser/             文档读取和编码回退
index/                 BM25、向量、融合、rerank
data/kb/docs/          随项目提供的测试语料
data/kb/questions.md   可编辑的评测题目集
tests/                 单元测试
eval_maxkb.py          与 Web 使用同一套评测逻辑的 CLI 入口
```

## 环境要求

- Python 3.11+
- DeepSeek API Key
- SiliconFlow API Key

API Key 只通过环境变量读取，不要写入代码、README 或提交历史。

PowerShell 示例：

```powershell
$env:DEEPSEEK_API_KEY = "your-deepseek-key"
$env:SILICONFLOW_API_KEY = "your-siliconflow-key"
```

## 安装

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 启动 Web 项目

```powershell
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

浏览器打开：http://127.0.0.1:8000

使用流程：

1. 左侧上传 `.md` / `.txt`，或导入本地目录。
2. 点击“重建索引”。首次构建会调用 SiliconFlow 生成向量。
3. 在输入框提问。回答中的 `[1]`、`[2]` 可以点击查看证据原文。
4. 评测题目集位于 `data/kb/questions.md`，也可以在网页中上传新的题目集。

## 题目集格式

题目集是普通 Markdown 文件，不需要修改 Python 代码：

```markdown
## 可答

1. 密码多久强制更换一次？ | 90 天
2. 试用期多长时间？ | 3 个月 | 三个月

## 拒答

1. 公司今年的年度团建地点是哪里？
```

可答题会检查关键词；拒答题会检查模型是否明确说明资料不足。匹配器会忽略空格、标点和大小写差异。

## CLI 评测

```powershell
.venv\Scripts\python.exe eval_maxkb.py
```

CLI 和 Web 使用同一套题目文件、语料和评测逻辑。评测会调用 DeepSeek，题目较多时会比普通提问耗时更长。

## 测试

```powershell
.venv\Scripts\python.exe -m pytest -q
```

## 隐私和提交范围

仓库包含示例测试语料和题目集，但不包含：

- API Keys、`.env` 文件或其他凭据
- Python 虚拟环境
- 本地向量索引、BM25 缓存和 DeepSeek 答案缓存
- NoMIRACL 大型运行时缓存

复制 `.env.example` 的变量到本机环境即可运行。公开仓库发布前，请仍然检查 Git 历史和 GitHub Secret Scanning。
