import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

class DocumentApiTests(unittest.TestCase):
    def test_real_chroma_and_api_lifecycle(self):
        import chromadb
        from fastapi.testclient import TestClient
        import api.main as api
        from rag.documents import DocumentStore
        from rag.ingestion import process_single_file
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            collection = chromadb.EphemeralClient().create_collection('integration_' + root.name.replace('-', '_'))
            self.addCleanup(lambda: collection.delete(where={'source_file': {'$in': ['a.txt', 'b.md']}}))
            store = DocumentStore(root/'raw', root/'chunks.json', collection, lambda texts: [[1., 0., .5] for _ in texts], process_single_file)
            with patch('rag.documents.get_store', return_value=store), patch.object(api, 'DATA_RAW_DIR', store.raw_dir), TestClient(api.app) as client:
                def upload(name, content):
                    return client.post('/upload', files={'file': (name, content, 'text/plain')})
                response = upload('a.txt', b'alpha facts')
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(upload('a.txt', b'alpha facts').json()['status'], 'unchanged')
                self.assertEqual(upload('b.md', b'beta facts').status_code, 200)
                self.assertEqual(upload('a.txt', b'revised facts').status_code, 200)
                self.assertEqual(collection.get(where={'source_file': 'a.txt'})['documents'], ['revised facts'])
                self.assertEqual(client.get('/documents/a.txt/content').content, b'revised facts')
                # A manually removed original must stop appearing in search after reindex.
                (store.raw_dir/'b.md').unlink()
                self.assertEqual(client.post('/reindex').status_code, 200)
                self.assertEqual(collection.get(where={'source_file': 'b.md'})['ids'], [])
                self.assertEqual(client.delete('/documents/a.txt').status_code, 200)
                self.assertEqual(collection.count(), 0)
                self.assertEqual(client.delete('/documents/a.txt').status_code, 404)
                self.assertEqual(upload('bad:stream.txt', b'bad').status_code, 400)
                self.assertEqual(upload('../bad.txt', b'bad').status_code, 400)

if __name__ == '__main__': unittest.main()
