import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION = "active_learning_kb"
DATA_DIR = Path(__file__).parent
DB_DIR = str(Path(__file__).parent / "chroma_data")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name=COLLECTION)

def index_articles():
    articles = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(DATA_DIR.glob("*.json"))]
    ids = [a["id"] for a in articles]
    texts = [a["title"] + ". " + a["content"] for a in articles]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()
    collection.upsert(ids=ids, documents=texts, embeddings=vectors,
                      metadatas=[{"title": a["title"], "category": a["category"]} for a in articles])
    return len(ids)

if __name__ == "__main__":
    print("Проиндексировано статей:", index_articles())
