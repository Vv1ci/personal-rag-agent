from rag.retrieval.retriever import retrieve
from rag.prompting.prompt_builder import build_rag_prompt


query = "What is attention in transformer models?"

retrieved_results = retrieve(query=query, top_k=3)

prompt = build_rag_prompt(
    query=query,
    retrieved_results=retrieved_results,
    language="zh"
)

print("===== RETRIEVED RESULTS =====")

for result in retrieved_results:
    print("=" * 60)
    print("rank:", result["rank"])
    print("id:", result["id"])
    print("relevance_score:", result["relevance_score"])
    print("metadata:", result["metadata"])
    print("text preview:", result["text"][:200])

print("\n===== RAG PROMPT =====")
print(prompt)