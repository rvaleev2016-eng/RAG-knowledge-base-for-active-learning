from retriever import search

TESTS = [
 ("Как выбрать объекты, в которых модель не уверена?", "uncertainty-01"),
 ("Как учесть разнообразие выбранных кандидатов?", "diversity-01"),
 ("Что делать после добавления новых размеченных объектов?", "iteration-01"),
 ("Когда остановить активное обучение?", "stopping-01"),
 ("Где хранятся кандидаты и итерации эксперимента?", "db-01"),
 ("Как ограничить стоимость эксперимента?", "budget-01"),
 ("Что нужно сохранять для воспроизводимости?", "repro-01"),
 ("Как развернуть сервис в контейнере?", "docker-01"),
]

def main():
    hits = 0
    reciprocal = 0.0
    for question, expected in TESTS:
        ids = [row["id"] for row in search(question, 3)]
        rank = ids.index(expected) + 1 if expected in ids else None
        hits += int(rank is not None)
        reciprocal += 1 / rank if rank else 0
        print(question, "->", ids, "rank:", rank)
    print(f"Recall@3={hits / len(TESTS):.4f}; MRR={reciprocal / len(TESTS):.4f}")

if __name__ == "__main__":
    main()
