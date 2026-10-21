
from sklearn.metrics.pairwise import cosine_similarity
import indexer


def search(query: str, k: int = 3):
    if indexer.matrix is None:
        return []

    k = max(1, min(k, len(indexer.articles)))

    query_vector = indexer.vectorizer.transform([query])
    similarities = cosine_similarity(
        query_vector, indexer.matrix
    )[0]

    best_indices = similarities.argsort()[::-1][:k]

    results = []

    for i in best_indices:
        article = indexer.articles[i]

        results.append({
            "id": article["id"],
            "title": article["title"],
            "text": article["title"] + ". " + article["content"],
            "distance": float(1 - similarities[i])
        })

    return results
