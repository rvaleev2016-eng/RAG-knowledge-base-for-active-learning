

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from indexer import index_articles, article_count
from retriever import search
from generator import answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    index_articles()
    yield


app = FastAPI(
    title="Active Learning RAG Assistant",
    description="Поиск и ответы по базе знаний об активном обучении",
    version="1.0.0",
    lifespan=lifespan
)


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=3, ge=1, le=15)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "articles": article_count()
    }


@app.post("/search")
def do_search(data: Query):
    return {
        "results": search(data.query, data.k)
    }


@app.post("/ask")
def do_ask(data: Query):
    return answer(data.query, data.k)
