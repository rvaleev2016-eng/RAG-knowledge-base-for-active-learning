from retriever import search

def answer(query: str, k: int = 3):
    matches = search(query, k)
    if not matches:
        return {"answer": "В базе знаний не найдено материалов.", "sources": []}
    # Учебный extractive-генератор: не обращается к внешней LLM.
    text = "По материалам базы знаний: " + " ".join(m["text"] for m in matches[:2])
    return {"answer": text, "sources": [m["id"] for m in matches]}
