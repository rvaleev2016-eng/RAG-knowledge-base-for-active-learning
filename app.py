from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from indexer import index_articles, collection
from retriever import search
from generator import answer

@asynccontextmanager
async def lifespan(app: FastAPI):
    index_articles()
    yield

app = FastAPI(title="Active Learning RAG Assistant", lifespan=lifespan)
class Query(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=3, ge=1, le=15)

@app.get("/health")
def health():
    return {"status": "ok", "articles": collection.count()}

@app.post("/search")
def do_search(data: Query):
    return {"results": search(data.query, data.k)}

@app.post("/ask")
def do_ask(data: Query):
    return answer(data.query, data.k)
