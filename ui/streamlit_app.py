import uuid
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag.config import settings
from urllib.parse import quote

import requests
import streamlit as st


DEFAULT_API_URL = (
    f"{settings.api_base_url}/agent"
)

DEFAULT_HEALTH_URL = (
    f"{settings.api_base_url}/health"
)

DEFAULT_UPLOAD_URL = (
    f"{settings.api_base_url}/upload"
)

DEFAULT_DOCUMENTS_URL = (
    f"{settings.api_base_url}/documents"
)


def check_api_health() -> tuple[bool, str]:
    """
    Check whether the FastAPI backend is running.

    Returns:
        tuple[bool, str]:
        - bool: True if backend is healthy, False otherwise
        - str: Message to show in the UI
    """

    try:
        response = requests.get(
            DEFAULT_HEALTH_URL,
            timeout=3,
        )

        if response.status_code == 200:
            data = response.json()

            return (
                True,
                "Backend is running: "
                f"{data.get('status', 'ok')}",
            )

        return (
            False,
            "Backend returned status code: "
            f"{response.status_code}",
        )

    except requests.exceptions.ConnectionError:
        return (
            False,
            "Cannot connect to FastAPI backend. "
            "Please start the API server first.",
        )

    except requests.exceptions.Timeout:
        return (
            False,
            "FastAPI backend health check timed out.",
        )

    except Exception as error:
        return (
            False,
            "Unexpected health check error: "
            f"{type(error).__name__}: {error}",
        )


def ask_rag_api(
    question: str,
    top_k: int,
    min_relevance_score: float,
    language: str,
    debug: bool,
    session_id: str,
    use_memory: bool,
) -> dict:
    """
    Send a question to the FastAPI /agent endpoint.

    Args:
        question:
            User question from the Streamlit text area.

        top_k:
            Number of chunks to retrieve.

        min_relevance_score:
            Minimum relevance score threshold.

        language:
            Answer language, "en" or "zh".

        debug:
            Whether to enable debug mode.

        session_id:
            Unique identifier for the current browser session.

        use_memory:
            Whether recent conversation turns should be used.

    Returns:
        dict:
            JSON response from the FastAPI backend.
    """

    payload = {
        "question": question,
        "top_k": top_k,
        "min_relevance_score": min_relevance_score,
        "language": language,
        "model": None,
        "debug": debug,
        "session_id": session_id,
        "use_memory": use_memory,
    }

    response = requests.post(
        DEFAULT_API_URL,
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "API request failed with status code "
            f"{response.status_code}: {response.text}"
        )

    return response.json()


def upload_document_api(
    uploaded_file,
    rebuild: bool,
) -> dict:
    """
    Upload a file to the FastAPI /upload endpoint.
    """

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type,
        )
    }

    response = requests.post(
        DEFAULT_UPLOAD_URL,
        files=files,
        params={
            "rebuild": str(rebuild).lower(),
        },
        timeout=600,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Upload failed with status code "
            f"{response.status_code}: {response.text}"
        )

    return response.json()


def get_documents_api() -> dict:
    """
    Get the current document list from FastAPI.
    """

    response = requests.get(
        DEFAULT_DOCUMENTS_URL,
        timeout=10,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Failed to get documents with status code "
            f"{response.status_code}: {response.text}"
        )

    return response.json()


def show_sources(
    sources: list[dict],
) -> None:
    """
    Display retrieved sources in the Streamlit page.

    Args:
        sources:
            A list of source dictionaries returned by the API.
    """

    if not sources:
        st.info("No sources returned.")
        return

    for index, source in enumerate(
        sources,
        start=1,
    ):
        name = source.get("source_file", "未知文档")
        page = source.get("page_number", -1)
        with st.expander(f"{index}. {name}" + (f" · 第 {page} 页" if page and page > 0 else "")):
            st.write(source.get("text", ""))
            st.link_button("打开原文件", f"{DEFAULT_DOCUMENTS_URL}/{quote(name, safe='')}/content")
            st.caption(f"检索相关性：{source.get('relevance_score', 0):.3f}（不是答案正确概率）")


def main() -> None:
    st.set_page_config(
        page_title="Personal RAG Agent",
        page_icon="📚",
        layout="wide",
    )

    # Create one unique ID for the current Streamlit browser session.
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(
            uuid.uuid4()
        )

    st.title("📚 Personal RAG Agent")

    st.caption(
        "A local document-based RAG assistant powered by "
        "FastAPI, ChromaDB, and Ollama."
    )

    is_healthy, health_message = check_api_health()

    if is_healthy:
        st.success(health_message)

    else:
        st.error(health_message)

        st.warning(
            "Start the FastAPI backend with: "
            "py -m uvicorn api.main:app --reload"
        )

    st.divider()

    # =================================================
    # Document upload
    # =================================================

    st.subheader("Upload documents")

    uploaded_file = st.file_uploader(
        "Upload a .txt, .md, or .pdf file",
        type=[
            "txt",
            "md",
            "pdf",
        ],
    )

    rebuild_after_upload = st.checkbox(
        "上传后索引此文档（同名更新，内容相同则跳过）",
        value=True,
    )

    upload_button = st.button(
        "Upload document"
    )

    if upload_button:
        if uploaded_file is None:
            st.warning(
                "Please choose a file before uploading."
            )

        elif not is_healthy:
            st.error(
                "FastAPI backend is not available. "
                "Please start it first."
            )

        else:
            with st.spinner(
                "Uploading file and rebuilding "
                "knowledge base..."
            ):
                try:
                    upload_result = (
                        upload_document_api(
                            uploaded_file=uploaded_file,
                            rebuild=rebuild_after_upload,
                        )
                    )

                except Exception as error:
                    st.error(
                        "Upload failed: "
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

                else:
                    st.success(
                        upload_result.get(
                            "message",
                            "Upload completed.",
                        )
                    )

                    st.json(upload_result)

    with st.expander(
        "Current documents in knowledge base"
    ):
        if is_healthy:
            try:
                documents_result = (
                    get_documents_api()
                )

                for document in documents_result.get("documents", []):
                    name = document["filename"]
                    st.write(f"{name} · {document['size_bytes']} 字节")
                    confirmed = st.checkbox("确认删除文件及其索引", key=f"confirm-{name}")
                    if st.button("删除", key=f"delete-{name}", disabled=not confirmed):
                        response = requests.delete(f"{DEFAULT_DOCUMENTS_URL}/{quote(name, safe='')}", timeout=60)
                        response.raise_for_status()
                        st.rerun()

            except Exception as error:
                st.error(
                    "Failed to load documents: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )

        else:
            st.info(
                "Start the FastAPI backend "
                "to view documents."
            )

    st.divider()

    # =================================================
    # Question answering
    # =================================================

    st.subheader("Ask a question")

    question = st.text_area(
        "Question",
        value=(
            "What is ChromaDB used for in RAG?"
        ),
        height=120,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=20,
            value=settings.default_top_k,
            step=1,
        )

    with col2:
        min_relevance_score = st.slider(
            "Minimum relevance score",
            min_value=0.0,
            max_value=1.0,
            value=settings.min_relevance_score,
            step=0.01,
        )

    with col3:
        language = st.selectbox(
            "Answer language",
            options=[
                "zh",
                "en",
            ],
            index=0,
        )

    # =================================================
    # Memory settings
    # =================================================

    use_memory = st.checkbox(
        "Use conversation memory",
        value=True,
        help=(
            "Use recent conversation turns to "
            "understand follow-up questions."
        ),
    )

    st.caption(
        "Current session: "
        + st.session_state.session_id[:8]
    )

    debug = st.checkbox(
        "Debug mode",
        value=False,
    )

    ask_button = st.button(
        "Ask RAG Agent",
        type="primary",
    )

    if ask_button:
        if not question.strip():
            st.warning(
                "Please enter a question before asking."
            )
            return

        if not is_healthy:
            st.error(
                "FastAPI backend is not available. "
                "Please start it first."
            )
            return

        with st.spinner(
            "Retrieving context and "
            "generating answer..."
        ):
            try:
                result = ask_rag_api(
                    question=question.strip(),
                    top_k=top_k,
                    min_relevance_score=(
                        min_relevance_score
                    ),
                    language=language,
                    debug=debug,
                    session_id=(
                        st.session_state.session_id
                    ),
                    use_memory=use_memory,
                )

            except Exception as error:
                st.error(
                    "Failed to get answer: "
                    f"{type(error).__name__}: "
                    f"{error}"
                )
                return

        st.divider()

        st.subheader("Answer")
        st.write(
            result.get(
                "answer",
                "",
            )
        )

        st.subheader("Sources")
        show_sources(
            result.get(
                "sources",
                [],
            )
        )

        st.subheader("Metadata")
        st.json(
            result.get(
                "metadata",
                {},
            )
        )


if __name__ == "__main__":
    main()