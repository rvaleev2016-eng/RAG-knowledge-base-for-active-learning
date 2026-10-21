
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = Path(__file__).parent

articles = []
vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer="char_wb",
    ngram_range=(2, 4)
)
matrix = None


def index_articles():
    global articles, matrix

    articles = []

    for path in sorted(DATA_DIR.glob("*.json")):
        article = json.loads(path.read_text(encoding="utf-8"))
        articles.append(article)

    if articles:
        texts = [
            article["title"] + ". " + article["content"]
            for article in articles
        ]
        matrix = vectorizer.fit_transform(texts)

    return len(articles)


def article_count():
    return len(articles)


if __name__ == "__main__":
    print("Проиндексировано статей:", index_articles())
