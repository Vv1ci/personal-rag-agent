import tempfile
import unittest
from pathlib import Path
from rag.documents import DocumentStore, safe_name

class Collection:
    def __init__(self): self.data = {}; self.fail = False
    def get(self, where, include):
        rows = [(k, v) for k, v in self.data.items() if v[1]['source_file'] == where['source_file']]
        return dict(ids=[r[0] for r in rows], documents=[r[1][0] for r in rows],
                    metadatas=[r[1][1] for r in rows], embeddings=[r[1][2] for r in rows])
    def delete(self, where):
        self.data = {k:v for k,v in self.data.items() if v[1]['source_file'] != where['source_file']}
    def upsert(self, ids, documents, metadatas, embeddings):
        if self.fail: self.fail = False; raise RuntimeError('storage unavailable')
        self.data.update({k:(d,m,e) for k,d,m,e in zip(ids,documents,metadatas,embeddings)})

class DocumentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name); self.calls = 0; self.db = Collection()
        def process(path):
            text = Path(path).read_text()
            if text == 'broken': raise ValueError('parse failed')
            return [dict(chunk_id=Path(path).name, text=text, metadata={'source_file':Path(path).name, 'page_number':None})]
        def embed(texts): self.calls += 1; return [[1.,2.] for text in texts]
        self.store = DocumentStore(root/'raw', root/'chunks.json', self.db, embed, process)
    def test_duplicate_update_delete(self):
        self.store.update('a.txt', b'first'); self.store.update('b.txt', b'other')
        self.assertEqual(self.store.update('a.txt', b'first')['status'], 'unchanged')
        self.assertEqual(self.calls, 2)
        self.store.update('a.txt', b'new')
        self.assertEqual(self.db.data['a.txt'][0], 'new')
        self.store.update('a.txt', delete=True)
        self.assertNotIn('a.txt', self.db.data); self.assertIn('b.txt', self.db.data)
        self.assertFalse((self.store.raw_dir/'a.txt').exists())
    def test_failure_restores_previous_source_and_index(self):
        self.store.update('a.txt', b'first'); self.db.fail = True
        with self.assertRaises(RuntimeError): self.store.update('a.txt', b'new')
        self.assertEqual((self.store.raw_dir/'a.txt').read_bytes(), b'first')
        self.assertEqual(self.db.data['a.txt'][0], 'first')
        with self.assertRaises(ValueError): self.store.update('a.txt', b'broken')
        self.assertEqual(self.db.data['a.txt'][0], 'first')
    def test_deferred_index_removes_stale_answer_source(self):
        self.store.update('a.txt', b'first')
        self.store.update('a.txt', b'new', index=False)
        self.assertNotIn('a.txt', self.db.data)
        self.store.update('a.txt', b'new')
        self.assertEqual(self.db.data['a.txt'][0], 'new')
    def test_path_traversal(self):
        for name in ['../a.txt', '..\\a.txt', 'folder/a.txt', 'a.exe']:
            with self.assertRaises(ValueError): safe_name(name)
