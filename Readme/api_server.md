# REST API (FastAPI)

Сервер поднимается командой:
```bash
sigla scripts serve --store <store_name> --port 8000 [--device cpu]
```

По умолчанию доступен на `http://localhost:8000`.

## Энд-поинты

| Метод | Путь              | Параметры                              | Описание |
|-------|-------------------|----------------------------------------|----------|
| GET   | `/search`         | `query`, `top_k`, `tags`               | Семантический поиск |
| GET   | `/ask`            | `query`, `top_k`, `tags`, `temperature`| MERGE результатов (контекст) |
| GET   | `/capsule/{id}`   | —                                      | Получить мета капсулы |
| POST  | `/update`         | BODY: `[capsule, …]`                   | Добавить новые капсулы |
| GET   | `/info`           | —                                      | Общая информация о стора |
| GET   | `/list`           | `limit`, `tags`                        | Список капсул (пагинация) |
| GET   | `/walk`           | `query`, `depth`, `limit`, `algo`, `restart` | BFS/RandomWalk расширение |
| GET   | `/compress`       | `query`, `top_k`, `model`              | Сжатие контекста (summarize) |
| POST  | `/prune`          | `ids`, `tags`                          | Удалить капсулы и переиндексировать |
| POST  | `/reindex`        | `model`, `factory`                     | Полный rebuild индекса |

### Пример запроса `/search`
```bash
curl "http://localhost:8000/search?query=vector+database&top_k=3"
```
Ответ:
```json
[
  {
    "id": 12,
    "text": "What is a vector database? …",
    "score": 0.92,
    "tags": ["db", "ml"]
  },
  …
]
```

> ⚠️  При включённой `API-Key` аутентификации добавьте заголовок `Authorization: Bearer <key>` (см. `sigla/server.py`).

---

OpenAPI-спека доступна по адресу `/docs` (Swagger UI) и `/openapi.json`. 