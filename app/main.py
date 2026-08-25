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


class DirRequest(BaseModel):
    directory: str


class DeleteRequest(BaseModel):
    document_id: str


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
        return kb.rebuild(force=True)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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
        return kb.ask(req.query)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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
