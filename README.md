# MiniRAG

一个可运行的小型中文 RAG 问答项目：文档解析、结构化分块、BM25 + BGE-M3 向量混合检索、RRF 融合、rerank，以及带来源引用的 DeepSeek 回答。

项目包含一个 FastAPI 后端和一个不需要 Node.js 构建的 HTML/CSS/JavaScript 单页前端。

## 功能

- 支持网页上传和本地目录导入 Markdown / TXT / PDF / DOCX / HTML 文档
- DOCX 标题、段落和表格以及 HTML 正文和表格统一转换为 Markdown；PDF 按页保留页码标题
- 结构化 Markdown 分块，保留标题路径和原文位置
- BM25 + SiliconFlow `BAAI/bge-m3` 向量检索
- RRF 融合和 rerank
- DeepSeek 带 `[n]` 引用回答
- 点击回答中的引用编号查看证据原文
- 文档指纹缓存索引，文档变化后自动失效
- SQLite 文档目录提供稳定文档 ID、SHA-256 内容去重和版本历史
- 同名同内容上传幂等；同名内容变化创建新版本；删除采用逻辑删除
- 检索证据返回文档 ID、版本 ID、Chunk ID 和原文偏移，便于审计追踪
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
data/kb/catalog.sqlite3 本地文档目录（运行时生成，不提交）
data/kb/blobs/         按内容哈希保存的不可变文档版本（运行时生成，不提交）
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

1. 左侧上传 `.md` / `.txt` / `.pdf` / `.docx` / `.html`，或导入本地目录。
2. 点击“重建索引”。首次构建会调用 SiliconFlow 生成向量。
3. 在输入框提问。回答中的 `[1]`、`[2]` 可以点击查看证据原文。
4. 评测题目集位于 `data/kb/questions.md`，也可以在网页中上传新的题目集。

同名文档再次上传时不会再生成 `_2` 副本：内容未变化则保持当前版本，内容变化则创建新版本。文档列表显示当前版本号；历史版本可通过 `GET /api/documents/{document_id}/versions` 查询。首次升级启动时，现有 `data/kb/docs/` 文档会自动登记为版本 1。

当前删除为逻辑删除：文档会从当前索引输入中排除，但版本记录和内容 Blob 会保留，为后续审计、恢复和 ACL 打基础。文档发生新增、更新或删除后，需要重建索引；索引发布前仍沿用上一个完整内存索引，后续阶段再实现按 Chunk 复用向量和原子索引切换。

当前 PDF 支持可直接提取文本的文件。纯扫描 PDF 会以明确错误拒绝入库，需要 OCR/VLM 的图片语义解析留在多模态阶段；旧 `.doc` 二进制格式暂不支持，请先转换为 `.docx`。DOCX 中的图片目前不进入文本索引，但标题、正文和表格会保留。

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
