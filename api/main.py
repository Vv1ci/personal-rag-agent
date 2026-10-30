from datetime import datetime
import inspect
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag.agent.agent_controller import run_agent
from rag.config import settings
from rag.logging_config import get_logger


app = FastAPI(
    title=settings.app_name,
    description="A local API wrapper for the Personal RAG Agent project.",
    version="0.2.0",
)

PROJECT_ROOT = settings.project_root
DATA_RAW_DIR = settings.data_raw_dir

ALLOWED_UPLOAD_EXTENSIONS = set(
    settings.allowed_upload_extensions
)

logger = get_logger(__name__)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )

    top_k: int = Field(
        default=settings.default_top_k,
        ge=1,
        le=20,
    )

    min_relevance_score: float = Field(
        default=settings.min_relevance_score,
        ge=0.0,
        le=1.0,
    )

    language: Literal["en", "zh"] = "zh"
    model: Optional[str] = None
    debug: bool = False

    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
    )

    use_memory: bool = True


class AskResponse(BaseModel):
    answer: str
    sources: list[dict] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


@app.get("/")
def root() -> dict:
    return {
        "message": f"{settings.app_name} API is running.",
        "docs": "/docs",
        "health": "/health",
        "ask": "/ask",
        "agent": "/agent",
    }


@app.post("/agent", response_model=AskResponse)
def agent_question(request: AskRequest) -> dict[str, Any]:
    try:
        logger.info(
            (
                "Agent request received | "
                "session=%s | "
                "question_length=%s | "
                "memory=%s"
            ),
            request.session_id,
            len(request.question),
            request.use_memory,
        )

        result = run_agent(
            question=request.question,
            top_k=request.top_k,
            language=request.language,
            min_relevance_score=(
                request.min_relevance_score
            ),
            debug=request.debug,
            session_id=request.session_id,
            use_memory=request.use_memory,
        )

        logger.info(
            (
                "Agent request completed | "
                "session=%s | "
                "route=%s | "
                "tool=%s"
            ),
            request.session_id,
            result.get("metadata", {}).get(
                "agent_route",
                "unknown",
            ),
            result.get("metadata", {}).get(
                "tool_used"
            ),
        )

        return result

    except Exception as exc:
        logger.exception(
            "Agent execution failed | session=%s",
            request.session_id,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Agent execution failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": f"{settings.app_name} API",
        "version": "0.2.0",
        "environment": settings.app_env,
        "time": datetime.now().isoformat(
            timespec="seconds"
        ),
    }


@app.post("/echo")
def echo(request: AskRequest) -> dict:
    return {
        "received_question": request.question,
        "received_top_k": request.top_k,
        "received_min_relevance_score": (
            request.min_relevance_score
        ),
        "received_language": request.language,
        "received_model": request.model,
        "received_debug": request.debug,
        "received_session_id": request.session_id,
        "received_use_memory": request.use_memory,
    }


def ensure_list_of_dicts(
    value: Any,
) -> list[dict]:
    """
    Convert sources into a safe list[dict] format.

    The RAG chain may return sources in different formats.
    This helper makes the API response stable.
    """

    if value is None:
        return []

    if isinstance(value, list):
        normalized_items = []

        for item in value:
            if isinstance(item, dict):
                normalized_items.append(item)
            else:
                normalized_items.append(
                    {"value": str(item)}
                )

        return normalized_items

    if isinstance(value, dict):
        return [value]

    return [{"value": str(value)}]


def normalize_rag_result(
    raw_result: Any,
) -> AskResponse:
    """
    Convert the output of answer_with_rag() into AskResponse.

    The RAG chain may return a dictionary, tuple, or string.
    """

    if isinstance(raw_result, AskResponse):
        return raw_result

    if isinstance(raw_result, dict):
        answer = raw_result.get("answer", "")
        sources = raw_result.get("sources", [])
        metadata = raw_result.get("metadata", {})

        if not answer:
            answer = str(raw_result)

        return AskResponse(
            answer=str(answer),
            sources=ensure_list_of_dicts(sources),
            metadata=(
                dict(metadata)
                if isinstance(metadata, dict)
                else {"metadata": str(metadata)}
            ),
        )

    if isinstance(raw_result, tuple):
        if len(raw_result) >= 2:
            answer = raw_result[0]
            sources = raw_result[1]

            metadata = {}

            if (
                len(raw_result) >= 3
                and isinstance(raw_result[2], dict)
            ):
                metadata = raw_result[2]

            return AskResponse(
                answer=str(answer),
                sources=ensure_list_of_dicts(
                    sources
                ),
                metadata=metadata,
            )

        if len(raw_result) == 1:
            return AskResponse(
                answer=str(raw_result[0])
            )

    return AskResponse(
        answer=str(raw_result),
        sources=[],
        metadata={
            "raw_result_type": (
                type(raw_result).__name__
            )
        },
    )


def call_rag_chain(
    request: AskRequest,
) -> Any:
    """
    Call the real RAG pipeline with a lazy import.
    """

    from rag.chains.rag_chain import answer_with_rag

    signature = inspect.signature(answer_with_rag)
    parameters = signature.parameters

    kwargs = {}

    question_parameter_names = [
        "query",
        "question",
        "user_query",
    ]

    question_is_bound = False

    for name in question_parameter_names:
        if name in parameters:
            kwargs[name] = request.question
            question_is_bound = True
            break

    candidate_kwargs = {
        "top_k": request.top_k,
        "k": request.top_k,
        "min_relevance_score": (
            request.min_relevance_score
        ),
        "threshold": request.min_relevance_score,
        "language": request.language,
        "debug": request.debug,
    }

    if request.model:
        candidate_kwargs["model"] = request.model
        candidate_kwargs["model_name"] = (
            request.model
        )

    for name, value in candidate_kwargs.items():
        if name in parameters:
            kwargs[name] = value

    if question_is_bound:
        return answer_with_rag(**kwargs)

    return answer_with_rag(
        request.question,
        **kwargs,
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        logger.info(
            (
                "Direct RAG request received | "
                "question_length=%s"
            ),
            len(request.question),
        )

        raw_result = call_rag_chain(request)
        response = normalize_rag_result(raw_result)

        response.metadata["api_mode"] = "real_rag"
        response.metadata["question"] = (
            request.question
        )
        response.metadata["top_k"] = request.top_k
        response.metadata[
            "min_relevance_score"
        ] = request.min_relevance_score
        response.metadata["language"] = (
            request.language
        )

        if request.model:
            response.metadata["model"] = (
                request.model
            )

        return response

    except Exception as error:
        logger.exception(
            "Direct RAG request failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "RAG pipeline failed: "
                f"{type(error).__name__}: {error}"
            ),
        ) from error


def make_safe_upload_path(
    filename: str,
) -> Path:
    """
    Create a safe path inside the configured raw-data directory.
    """

    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()

    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(
            sorted(ALLOWED_UPLOAD_EXTENSIONS)
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {suffix}. "
                f"Allowed types: {allowed}"
            ),
        )

    DATA_RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_path = DATA_RAW_DIR / safe_name

    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix

    counter = 1

    while True:
        candidate = (
            DATA_RAW_DIR
            / f"{stem}_{counter}{suffix}"
        )

        if not candidate.exists():
            return candidate

        counter += 1


def list_raw_documents() -> list[dict]:
    """
    List files currently stored in the configured raw-data directory.
    """

    DATA_RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents = []

    for path in sorted(DATA_RAW_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_UPLOAD_EXTENSIONS:
            continue

        documents.append(
            {
                "filename": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
        )

    return documents


def run_reindex_pipeline() -> dict:
    from rag.documents import get_store, INDEX_LOCK
    import json
    store = get_store()
    with INDEX_LOCK:
        DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        present = {path.name for path in DATA_RAW_DIR.iterdir() if path.is_file() and path.suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS}
        metadata = store.collection.get(include=['metadatas'])['metadatas']
        indexed = {item['source_file'] for item in metadata if item.get('source_file')}
        if store.chunks_path.exists():
            indexed.update(c['metadata']['source_file'] for c in json.loads(store.chunks_path.read_text(encoding='utf-8')))
        removed = [store.update(name, delete=True, allow_missing=True) for name in sorted(indexed - present)]
        results = [store.update(path.name, path.read_bytes())
                   for path in sorted(DATA_RAW_DIR.iterdir())
                   if path.is_file() and path.suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS]
    return {"status": "ok", "documents": results, "removed": removed}


@app.get("/documents")
def documents() -> dict:
    """
    Return documents currently stored in data/raw.
    """

    return {
        "documents": list_raw_documents(),
        "raw_dir": str(DATA_RAW_DIR),
    }


@app.post("/reindex")
def reindex() -> dict:
    """
    Manually rebuild chunks and ChromaDB.
    """

    try:
        result = run_reindex_pipeline()

        logger.info(
            "Knowledge base reindexed successfully"
        )

        return {
            "message": (
                "Knowledge base reindexed successfully."
            ),
            "reindex": result,
            "documents": list_raw_documents(),
        }

    except Exception as error:
        logger.exception(
            "Knowledge-base reindex failed"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Reindex failed: "
                f"{type(error).__name__}: {error}"
            ),
        ) from error


@app.post("/upload")
def upload_document(file: UploadFile = File(...), rebuild: bool = True) -> dict:
    from rag.documents import get_store, MAX_UPLOAD_BYTES
    try:
        content = file.file.read(MAX_UPLOAD_BYTES + 1)
        result = get_store().update(file.filename or "", content, index=rebuild)
        return {"message": "文档已处理", **result, "documents": list_raw_documents()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document update failed")
        raise HTTPException(status_code=500, detail="文档处理失败，原文档与索引已尝试恢复，请查看日志") from exc


@app.delete("/documents/{filename}")
def delete_document(filename: str) -> dict:
    from rag.documents import get_store
    try:
        return get_store().update(filename, delete=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="文档不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/documents/{filename}/content")
def document_content(filename: str):
    from rag.documents import safe_name
    from fastapi.responses import FileResponse
    try:
        path = DATA_RAW_DIR / safe_name(filename)
        if path.resolve().parent != DATA_RAW_DIR.resolve():
            raise ValueError('文档路径不在知识库内')
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文档不存在")
    return FileResponse(path, filename=path.name)
