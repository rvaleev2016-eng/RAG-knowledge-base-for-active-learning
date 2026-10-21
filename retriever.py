from indexer import collection, model

def search(query: str, k: int = 3):
    k = max(1, min(k, 15))
    vector = model.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=vector, n_results=k,
                              include=["documents", "metadatas", "distances"])
    return [{"id": ident, "text": text, "title": meta["title"], "distance": distance}
            for ident, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                                   result["metadatas"][0], result["distances"][0])]
