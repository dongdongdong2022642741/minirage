"""MiniRAG Web: FastAPI 后端（多知识库版）。

每个知识库是独立数据根目录；请求携带 kb_id 路由到对应实例，
缺省为 main。挂载清单见 data/kbs/registry.json。

Run:  cd minirage && .venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.kb import KnowledgeBase
from app.document_catalog import DocumentNotFoundError, DocumentStateError
from app.kb_registry import DEFAULT_KB_ID, KBRegistry

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
kb = KnowledgeBase(ROOT / "data" / "kb")  # 默认库（兼容既有调用方/测试）
REGISTRY = KBRegistry(ROOT / "data" / "kbs", ROOT / "data" / "kb")

app = FastAPI(title="MiniRAG 知识库问答", version="0.3.0")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


def resolve_kb(kb_id: str | None) -> KnowledgeBase:
    if not kb_id or kb_id == DEFAULT_KB_ID:
        return kb
    try:
        return REGISTRY.get(kb_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


class AskRequest(BaseModel):
    query: str
    user_id: str
    kb_id: str = ""


class DirRequest(BaseModel):
    directory: str
    kb_id: str = ""


class DeleteRequest(BaseModel):
    document_id: str
    kb_id: str = ""


class AclRequest(BaseModel):
    user_id: str
    document_id: str
    allow: bool
    kb_id: str = ""


class UserRequest(BaseModel):
    user_id: str
    display_name: str = ""
    kb_id: str = ""


class KbCreateRequest(BaseModel):
    kb_id: str
    name: str
    description: str = ""


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE / "static" / "index.html")


# ---------- 知识库挂载 ----------

@app.get("/api/kbs")
def list_kbs() -> dict:
    return {"kbs": REGISTRY.list(), "default": DEFAULT_KB_ID}


@app.post("/api/kbs")
def create_kb(req: KbCreateRequest) -> dict:
    try:
        entry = REGISTRY.create(req.kb_id, req.name, req.description)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"kb": entry, "kbs": REGISTRY.list()}


# ---------- 状态与文档 ----------

@app.get("/api/status")
def status(kb_id: str = "") -> dict:
    return resolve_kb(kb_id).status()


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...),
                 kb_id: str = Form("")) -> dict:
    svc = resolve_kb(kb_id)
    try:
        saved = svc.add_uploads([(f.filename or "", await f.read()) for f in files])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"saved": saved}


@app.post("/api/import_dir")
def import_dir(req: DirRequest) -> dict:
    svc = resolve_kb(req.kb_id)
    try:
        imported = svc.import_dir(req.directory)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"imported": imported}


@app.post("/api/delete")
def delete(req: DeleteRequest) -> dict:
    return {"deleted": resolve_kb(req.kb_id).delete_doc(req.document_id)}


@app.get("/api/documents/{document_id}/versions")
def document_versions(document_id: str, kb_id: str = "") -> dict:
    return {"versions": resolve_kb(kb_id).document_versions(document_id)}


@app.post("/api/documents/{document_id}/restore")
def restore_document(document_id: str, kb_id: str = "") -> dict:
    try:
        return resolve_kb(kb_id).restore_document(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DocumentStateError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


# ---------- 构建 ----------

@app.post("/api/rebuild")
def rebuild(kb_id: str = "") -> dict:
    try:
        return resolve_kb(kb_id).start_rebuild()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/builds/current")
def build_current(kb_id: str = "") -> dict:
    job = resolve_kb(kb_id).current_build()
    if job is None:
        raise HTTPException(status_code=404, detail="当前没有构建任务")
    return job


@app.get("/api/builds")
def build_history(limit: int = 10, kb_id: str = "") -> dict:
    svc = resolve_kb(kb_id)
    return {"jobs": svc.catalog.recent_build_jobs(limit=max(1, min(limit, 50)))}


# ---------- 运维 ----------

@app.get("/api/stats")
def stats(kb_id: str = "") -> dict:
    return resolve_kb(kb_id).opstats()


@app.get("/api/audit")
def audit_tail(n: int = 30, kb_id: str = "") -> dict:
    return {"events": resolve_kb(kb_id).audit.tail(n=max(1, min(n, 200)))}


# ---------- 问答 ----------

@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    try:
        return resolve_kb(req.kb_id).ask(req.query, req.user_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


# ---------- ACL ----------

@app.get("/api/users")
def users(kb_id: str = "") -> dict:
    return {"users": resolve_kb(kb_id).catalog.list_users()}


@app.post("/api/users")
def create_user(req: UserRequest) -> dict:
    svc = resolve_kb(req.kb_id)
    try:
        svc.catalog.ensure_user(req.user_id, req.display_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    svc.audit.record("user_created", user_id=req.user_id)
    return {"users": svc.catalog.list_users()}


@app.post("/api/acl")
def update_acl(req: AclRequest) -> dict:
    svc = resolve_kb(req.kb_id)
    try:
        if req.allow:
            svc.catalog.grant(req.user_id, req.document_id)
        else:
            changed = svc.catalog.revoke(req.user_id, req.document_id)
            if not changed:
                raise ValueError("该授权不存在")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    svc.audit.record("acl", user_id=req.user_id,
                     document_id=req.document_id, allowed=req.allow)
    return {"user_id": req.user_id, "document_id": req.document_id, "allowed": req.allow}


# ---------- 题目集与评测 ----------

@app.get("/api/questions")
def questions(kb_id: str = "") -> dict:
    try:
        return resolve_kb(kb_id).questions_info()
    except ValueError as error:
        return {"exists": False, "detail": str(error)}


@app.post("/api/questions_upload")
async def questions_upload(file: UploadFile = File(...), kb_id: str = Form("")) -> dict:
    try:
        return resolve_kb(kb_id).upload_questions(file.filename or "", await file.read())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/eval")
def eval_questions(kb_id: str = "") -> dict:
    try:
        return resolve_kb(kb_id).run_eval()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.exception_handler(HTTPException)
def http_error_handler(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
