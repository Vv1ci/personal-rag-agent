from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


# document_stats_tool.py 位于：
# project/rag/agent/tools/document_stats_tool.py
#
# parents[0] = tools
# parents[1] = agent
# parents[2] = rag
# parents[3] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "all_chunks.json"
)

SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".pdf",
}


def _load_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    """
    Load chunks from all_chunks.json.

    Supported JSON structures:

    1. A list:
       [
           {"text": "...", "metadata": {...}}
       ]

    2. A dictionary containing a chunks list:
       {
           "chunks": [
               {"text": "...", "metadata": {...}}
           ]
       }
    """

    if not chunks_path.exists():
        return []

    try:
        file_content = chunks_path.read_text(
            encoding="utf-8-sig"
        )
        json_data = json.loads(file_content)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON file: {chunks_path}"
        ) from exc

    if isinstance(json_data, list):
        raw_chunks = json_data

    elif (
        isinstance(json_data, dict)
        and isinstance(json_data.get("chunks"), list)
    ):
        raw_chunks = json_data["chunks"]

    else:
        raise ValueError(
            "all_chunks.json must contain a list of chunks "
            "or a dictionary with a 'chunks' list."
        )

    return [
        chunk
        for chunk in raw_chunks
        if isinstance(chunk, dict)
    ]


def _normalise_source_name(value: Any) -> str | None:
    """
    Convert a path or source value into a simple filename.

    Example:
    data/raw/example.txt -> example.txt
    C:\\project\\data\\raw\\example.txt -> example.txt
    """

    if not isinstance(value, str):
        return None

    cleaned_value = value.strip()

    if not cleaned_value:
        return None

    normalised_path = cleaned_value.replace("\\", "/")

    return normalised_path.rsplit("/", 1)[-1]


def _extract_source_name(
    chunk: dict[str, Any],
) -> str | None:
    """
    Extract the source filename from different possible chunk formats.
    """

    metadata = chunk.get("metadata")

    if not isinstance(metadata, dict):
        metadata = {}

    possible_sources = [
        chunk.get("source"),
        chunk.get("file_name"),
        chunk.get("filename"),
        chunk.get("path"),
        metadata.get("source"),
        metadata.get("file_name"),
        metadata.get("filename"),
        metadata.get("path"),
    ]

    for source_value in possible_sources:
        source_name = _normalise_source_name(source_value)

        if source_name:
            return source_name

    return None


def _extract_chunk_text(
    chunk: dict[str, Any],
) -> str:
    """
    Extract chunk text from several common field names.
    """

    possible_text_fields = [
        "text",
        "content",
        "page_content",
        "chunk_text",
    ]

    for field_name in possible_text_fields:
        value = chunk.get(field_name)

        if isinstance(value, str):
            return value

    return ""


def _list_raw_documents(
    raw_dir: Path,
) -> list[str]:
    """
    Return supported documents currently stored in data/raw.
    """

    if not raw_dir.exists():
        return []

    return sorted(
        file_path.name
        for file_path in raw_dir.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )


def build_document_statistics(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
) -> dict[str, Any]:
    """
    Build statistics by comparing data/raw with all_chunks.json.
    """

    raw_documents = _list_raw_documents(raw_dir)
    chunks = _load_chunks(chunks_path)

    chunk_counts: Counter[str] = Counter()
    character_counts: Counter[str] = Counter()

    chunks_without_source = 0

    for chunk in chunks:
        source_name = _extract_source_name(chunk)

        if source_name is None:
            chunks_without_source += 1
            continue

        chunk_text = _extract_chunk_text(chunk)

        chunk_counts[source_name] += 1
        character_counts[source_name] += len(chunk_text)

    raw_document_set = set(raw_documents)
    indexed_document_set = set(chunk_counts)

    # Raw files that have not produced any indexed chunks.
    unindexed_documents = sorted(
        raw_document_set - indexed_document_set
    )

    # Sources that still exist in all_chunks.json but no longer exist
    # inside data/raw.
    orphaned_index_sources = sorted(
        indexed_document_set - raw_document_set
    )

    all_document_names = sorted(
        raw_document_set | indexed_document_set
    )

    document_details: list[dict[str, Any]] = []

    for document_name in all_document_names:
        document_chunk_count = chunk_counts[document_name]
        document_character_count = (
            character_counts[document_name]
        )

        average_chunk_length = (
            document_character_count / document_chunk_count
            if document_chunk_count > 0
            else 0.0
        )

        document_details.append(
            {
                "name": document_name,
                "exists_in_raw": (
                    document_name in raw_document_set
                ),
                "is_indexed": (
                    document_name in indexed_document_set
                ),
                "chunk_count": document_chunk_count,
                "character_count": document_character_count,
                "average_chunk_length": round(
                    average_chunk_length,
                    2,
                ),
            }
        )

    indexed_document_count = len(indexed_document_set)
    total_chunk_count = sum(chunk_counts.values())

    average_chunks_per_document = (
        total_chunk_count / indexed_document_count
        if indexed_document_count > 0
        else 0.0
    )

    return {
        "raw_document_count": len(raw_documents),
        "indexed_document_count": indexed_document_count,
        "total_chunk_count": total_chunk_count,
        "average_chunks_per_document": round(
            average_chunks_per_document,
            2,
        ),
        "chunks_without_source": chunks_without_source,
        "unindexed_documents": unindexed_documents,
        "orphaned_index_sources": orphaned_index_sources,
        "documents": document_details,
        "raw_dir": str(raw_dir),
        "chunks_path": str(chunks_path),
    }


def _format_name_list(names: list[str]) -> str:
    if not names:
        return "None"

    return ", ".join(names)


def _format_summary_answer(
    statistics: dict[str, Any],
    language: str,
) -> str:
    """
    Format whole-knowledge-base statistics.
    """

    ranked_documents = sorted(
        statistics["documents"],
        key=lambda document: (
            -document["chunk_count"],
            document["name"].lower(),
        ),
    )

    document_lines = [
        (
            f"- {document['name']}: "
            f"{document['chunk_count']} chunks"
        )
        for document in ranked_documents[:10]
    ]

    if not document_lines:
        document_lines = ["- No indexed documents"]

    if language == "zh":
        return "\n".join(
            [
                "知识库统计",
                (
                    f"- 原始文档数量："
                    f"{statistics['raw_document_count']}"
                ),
                (
                    f"- 已索引文档数量："
                    f"{statistics['indexed_document_count']}"
                ),
                (
                    f"- 总 chunk 数量："
                    f"{statistics['total_chunk_count']}"
                ),
                (
                    f"- 平均每个文档 chunk 数："
                    f"{statistics['average_chunks_per_document']}"
                ),
                (
                    f"- 缺少 source 的 chunks："
                    f"{statistics['chunks_without_source']}"
                ),
                (
                    "- 尚未索引的原始文件："
                    f"{_format_name_list(
                        statistics['unindexed_documents']
                    )}"
                ),
                (
                    "- 仅存在于索引中的来源："
                    f"{_format_name_list(
                        statistics['orphaned_index_sources']
                    )}"
                ),
                "",
                "文档 chunk 统计：",
                *document_lines,
            ]
        )

    return "\n".join(
        [
            "Knowledge Base Statistics",
            (
                f"- Raw documents: "
                f"{statistics['raw_document_count']}"
            ),
            (
                f"- Indexed documents: "
                f"{statistics['indexed_document_count']}"
            ),
            (
                f"- Total chunks: "
                f"{statistics['total_chunk_count']}"
            ),
            (
                f"- Average chunks per document: "
                f"{statistics['average_chunks_per_document']}"
            ),
            (
                f"- Chunks without source metadata: "
                f"{statistics['chunks_without_source']}"
            ),
            (
                "- Unindexed raw documents: "
                f"{_format_name_list(
                    statistics['unindexed_documents']
                )}"
            ),
            (
                "- Indexed sources missing from data/raw: "
                f"{_format_name_list(
                    statistics['orphaned_index_sources']
                )}"
            ),
            "",
            "Document chunk counts:",
            *document_lines,
        ]
    )


def _resolve_document(
    target: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Resolve an exact or partial filename.
    """

    normalised_target = _normalise_source_name(target)

    if not normalised_target:
        raise ValueError("The document name cannot be empty.")

    target_lower = normalised_target.lower()

    exact_matches = [
        document
        for document in documents
        if document["name"].lower() == target_lower
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    partial_matches = [
        document
        for document in documents
        if target_lower in document["name"].lower()
    ]

    if len(partial_matches) == 1:
        return partial_matches[0]

    if len(partial_matches) > 1:
        matching_names = [
            document["name"]
            for document in partial_matches
        ]

        raise ValueError(
            "The document name is ambiguous. Matches: "
            + ", ".join(matching_names)
        )

    raise ValueError(
        f"Document not found: {normalised_target}"
    )


def _format_document_answer(
    document: dict[str, Any],
    language: str,
) -> str:
    if language == "zh":
        return "\n".join(
            [
                f"文档统计：{document['name']}",
                (
                    "- 存在于 data/raw："
                    f"{'是' if document['exists_in_raw'] else '否'}"
                ),
                (
                    "- 已进入索引："
                    f"{'是' if document['is_indexed'] else '否'}"
                ),
                f"- chunk 数量：{document['chunk_count']}",
                (
                    f"- chunk 字符总数："
                    f"{document['character_count']}"
                ),
                (
                    f"- 平均 chunk 长度："
                    f"{document['average_chunk_length']}"
                ),
            ]
        )

    return "\n".join(
        [
            f"Document Statistics: {document['name']}",
            (
                "- Present in data/raw: "
                f"{document['exists_in_raw']}"
            ),
            (
                "- Indexed: "
                f"{document['is_indexed']}"
            ),
            f"- Chunks: {document['chunk_count']}",
            (
                f"- Total chunk characters: "
                f"{document['character_count']}"
            ),
            (
                f"- Average chunk length: "
                f"{document['average_chunk_length']}"
            ),
        ]
    )


def run_document_stats_tool(
    target: str = "",
    *,
    language: str = "en",
    raw_dir: Path = DEFAULT_RAW_DIR,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
) -> dict[str, Any]:
    """
    Public interface used by the Agent Controller.

    Empty target:
        return whole-knowledge-base statistics

    Non-empty target:
        return statistics for one document
    """

    try:
        statistics = build_document_statistics(
            raw_dir=raw_dir,
            chunks_path=chunks_path,
        )

        cleaned_target = target.strip()

        if not cleaned_target:
            return {
                "success": True,
                "answer": _format_summary_answer(
                    statistics,
                    language,
                ),
                "data": statistics,
                "error": None,
            }

        document = _resolve_document(
            cleaned_target,
            statistics["documents"],
        )

        return {
            "success": True,
            "answer": _format_document_answer(
                document,
                language,
            ),
            "data": document,
            "error": None,
        }

    except (OSError, ValueError) as exc:
        return {
            "success": False,
            "answer": None,
            "data": None,
            "error": str(exc),
        }