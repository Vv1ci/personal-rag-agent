"""Single-process document updates; readers share the same lock.

Embeddings are computed before mutation. On a failed commit the previous index,
source file and chunk metadata are restored. Run the API with ONE worker.
"""
import hashlib
import json
import os
import re
from pathlib import Path
from threading import RLock

INDEX_LOCK = RLock()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def safe_name(name):
    if (not name or name != name.replace('\\', '/').split('/')[-1]
            or re.search(r'[<>:"|?*\x00-\x1f]', name) or name.endswith((' ', '.'))
            or name.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}):
        raise ValueError('请使用不包含目录的文件名')
    if Path(name).suffix.lower() not in {'.txt', '.md', '.pdf'}:
        raise ValueError('仅支持 TXT、Markdown 和文本型 PDF')
    return name


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_bytes(data)
    os.replace(temporary, path)


class DocumentStore:
    def __init__(self, raw_dir, chunks_path, collection, embed, process):
        self.raw_dir = Path(raw_dir)
        self.chunks_path = Path(chunks_path)
        self.collection = collection
        self.embed = embed
        self.process = process

    def update(self, name, content=None, *, delete=False, index=True, allow_missing=False):
        name = safe_name(name)
        if content is not None and len(content) > MAX_UPLOAD_BYTES:
            raise ValueError('文件不能超过 20 MB')
        with INDEX_LOCK:
            target = self.raw_dir / name
            if target.resolve().parent != self.raw_dir.resolve():
                raise ValueError('文档路径不在知识库内')
            previous = target.read_bytes() if target.exists() else None
            if delete and previous is None and not allow_missing:
                raise FileNotFoundError(name)
            old_json = self.chunks_path.read_bytes() if self.chunks_path.exists() else b'[]'
            chunks = json.loads(old_json)
            unchanged = previous == content and not delete
            digest = hashlib.sha256(content).hexdigest() if content is not None else None
            source_chunks = [c for c in chunks if c['metadata'].get('source_file') == name]
            old_vectors = self.collection.get(where={'source_file': name}, include=['documents', 'metadatas', 'embeddings'])
            indexed = source_chunks and set(old_vectors['ids']) == {c['chunk_id'] for c in source_chunks}
            if unchanged and ((not index and not old_vectors['ids'] and not source_chunks) or index and indexed and all(c['metadata'].get('content_sha256') == digest for c in source_chunks)):
                return {'filename': name, 'status': 'unchanged', 'chunks': len(source_chunks)}

            # Stage parsing without changing the user's original file.
            import tempfile
            replacement = []
            vectors = []
            if not delete and index:
                with tempfile.TemporaryDirectory() as directory:
                    staged = Path(directory) / name
                    staged.write_bytes(content)
                    replacement = self.process(str(staged))
                if not replacement:
                    raise ValueError('未提取到文本；扫描 PDF 需要先做 OCR')
                for chunk in replacement:
                    chunk['metadata']['content_sha256'] = digest
                vectors = self.embed([c['text'] for c in replacement])

            kept = [c for c in chunks if c['metadata'].get('source_file') != name]
            try:
                self.collection.delete(where={'source_file': name})
                if replacement:
                    self.collection.upsert(
                        ids=[c['chunk_id'] for c in replacement],
                        documents=[c['text'] for c in replacement],
                        metadatas=[{k: (-1 if v is None else v) for k, v in c['metadata'].items()} for c in replacement],
                        embeddings=vectors,
                    )
                if delete:
                    target.unlink(missing_ok=allow_missing)
                else:
                    atomic_write(target, content)
                atomic_write(self.chunks_path, json.dumps(kept + replacement, ensure_ascii=False, indent=2).encode())
            except Exception:
                self.collection.delete(where={'source_file': name})
                if old_vectors['ids']:
                    self.collection.upsert(**{k: old_vectors[k] for k in ['ids', 'documents', 'metadatas', 'embeddings']})
                if previous is None:
                    target.unlink(missing_ok=True)
                else:
                    atomic_write(target, previous)
                atomic_write(self.chunks_path, old_json)
                raise
            return {'filename': name, 'status': 'deleted' if delete else 'indexed' if index else 'saved', 'chunks': len(replacement)}


def get_store():
    from rag.config import settings
    from rag.ingestion import process_single_file
    from rag.vectorstore.chroma_store import get_chroma_collection
    from rag.embeddings.local_embedding import load_embedding_model, embed_texts
    return DocumentStore(settings.data_raw_dir, settings.chunks_path,
                         get_chroma_collection(),
                         lambda texts: embed_texts(texts, load_embedding_model()), process_single_file)
