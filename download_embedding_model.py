from rag.embeddings.local_embedding import load_embedding_model


model = load_embedding_model(local_files_only=False)

print("Embedding model cached and ready for offline use")
