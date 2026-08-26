"""MiniRAG Web: FastAPI 后端。

Run:  cd minirage && .venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.kb import KnowledgeBase
from app.document_catalog import DocumentNotFoundError, DocumentStateError

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
kb = KnowledgeBase(ROOT / "data" / "kb")

app = FastAPI(title="MiniRAG 知识库问答", version="0.2.0")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


class AskRequest(BaseModel):
    query: str
    user_id: str


class DirRequest(BaseModel):
    directory: str


class DeleteRequest(BaseModel):
    document_id: str


class AclRequest(BaseModel):
    user_id: str
    document_id: str
    allow: bool


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/status")
def status() -> dict:
    return kb.status()


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    try:
        saved = kb.add_uploads([(f.filename or "", await f.read()) for f in files])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"saved": saved}


@app.post("/api/import_dir")
def import_dir(req: DirRequest) -> dict:
    try:
        imported = kb.import_dir(req.directory)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"imported": imported}


@app.post("/api/rebuild")
def rebuild() -> dict:
    try:
        return kb.start_rebuild()
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/builds/current")
def build_current() -> dict:
    job = kb.current_build()
    if job is None:
        raise HTTPException(status_code=404, detail="当前没有构建任务")
    return job


@app.get("/api/builds")
def build_history(limit: int = 10) -> dict:
    return {"jobs": kb.catalog.recent_build_jobs(limit=max(1, min(limit, 50)))}


@app.get("/api/stats")
def stats() -> dict:
    return kb.opstats()


@app.get("/api/audit")
def audit_tail(n: int = 30) -> dict:
    return {"events": kb.audit.tail(n=max(1, min(n, 200)))}


@app.post("/api/delete")
def delete(req: DeleteRequest) -> dict:
    return {"deleted": kb.delete_doc(req.document_id)}


@app.get("/api/documents/{document_id}/versions")
def document_versions(document_id: str) -> dict:
    return {"versions": kb.document_versions(document_id)}


@app.post("/api/documents/{document_id}/restore")
def restore_document(document_id: str) -> dict:
    try:
        return kb.restore_document(document_id)
    except DocumentNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DocumentStateError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    try:
        return kb.ask(req.query, req.user_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/users")
def users() -> dict:
    return {"users": kb.catalog.list_users()}


class UserRequest(BaseModel):
    user_id: str
    display_name: str = ""


@app.post("/api/users")
def create_user(req: UserRequest) -> dict:
    try:
        kb.catalog.ensure_user(req.user_id, req.display_name)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    kb.audit.record("user_created", user_id=req.user_id)
    return {"users": kb.catalog.list_users()}


@app.post("/api/acl")
def update_acl(req: AclRequest) -> dict:
    try:
        if req.allow:
            kb.catalog.grant(req.user_id, req.document_id)
        else:
            changed = kb.catalog.revoke(req.user_id, req.document_id)
            if not changed:
                raise ValueError("该授权不存在")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    kb.audit.record("acl", user_id=req.user_id,
                    document_id=req.document_id, allowed=req.allow)
    return {"user_id": req.user_id, "document_id": req.document_id, "allowed": req.allow}


@app.get("/api/questions")
def questions() -> dict:
    try:
        return kb.questions_info()
    except ValueError as error:
        return {"exists": False, "detail": str(error)}


@app.post("/api/questions_upload")
async def questions_upload(file: UploadFile = File(...)) -> dict:
    try:
        return kb.upload_questions(file.filename or "", await file.read())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/eval")
def eval_questions() -> dict:
    try:
        return kb.run_eval()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.exception_handler(HTTPException)
def http_error_handler(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
