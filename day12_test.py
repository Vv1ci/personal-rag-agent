from rag.chains.rag_chain import answer_with_rag


queries = [
    "Transformer 里的 attention 是什么？",
    "embedding 是什么？",
    "ChromaDB 在 RAG 中有什么作用？"
]

for query in queries:
    print("=" * 80)
    print("QUERY:", query)

    result = answer_with_rag(
        query=query,
        top_k=3,
        language="zh"
    )

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for source in result["sources"]:
        print(
            f"- rank {source['rank']} | "
            f"{source['source_file']} | "
            f"page {source['page_number']} | "
            f"score {source['relevance_score']:.4f}"
        )

    print()