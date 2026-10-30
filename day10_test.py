from rag.retrieval.retriever import retrieve, build_context_from_results


queries = [
    "What is attention in transformer models?",
    "What is embedding?",
    "What did we learn about RAG and ChromaDB?"
]

for query in queries:
    print("=" * 80)
    print("QUERY:", query)

    results = retrieve(query=query, top_k=3)

    for result in results:
        print("-" * 60)
        print("rank:", result["rank"])
        print("id:", result["id"])
        print("distance:", result["distance"])
        print("relevance_score:", result["relevance_score"])
        print("source_file:", result["metadata"].get("source_file"))
        print("page_number:", result["metadata"].get("page_number"))
        print("text preview:", result["text"][:200])

    context = build_context_from_results(results)

    print("\nCONTEXT PREVIEW:")
    print(context[:500])
    print()