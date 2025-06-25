# Архитектура SIGLA

```
User → CLI / REST  ─┐
                    │  JSON (capsule)    ┌────────────┐
                    └──► CapsuleStore ───►  MetaStore │
                                        │ (SQLite…) │
LLM (HF) ◄─ Embeddings ◄─────────────────┘────────────┘
```

> *Стрелки слева направо — запись данных; справа налево — чтение/запросы.*

## 1. Слои

| Слой            | Описание | Пакеты/модули |
|-----------------|----------|---------------|
| Интерфейс       | CLI (`sigla scripts`), REST (`sigla.server`) | `sigla/scripts.py`, `sigla/server.py` |
| Логика          | Капсулы, поиск, DSL | `sigla/core.py`, `sigla/dsl.py`, `sigla/graph.py` |
| Метаданные      | Плагины для хранения meta (`MetaStore`) | `sigla/meta/*` |
| ML/Embeddings   | HF `SentenceTransformer` или локальный `TransformersEmbeddings` | `sigla/core.TransformersEmbeddings` |
| Хранение векторов | FAISS index (Flat / IVF / IVF+PQ) | внешняя зависимость `faiss` |

## 2. Потоки данных

### Инжест
1. CLI читает файлы → формирует список капсул `{text, tags}`.
2. `CapsuleStore.add_capsules` кодирует тексты в вектора.
3. Вектора добавляются в FAISS, мета — в `MetaStore`.
4. При необходимости `auto_link_k` создаёт k ближайших связей.

### Поиск
1. Вводной запрос кодируется той же моделью.
2. FAISS выполняет `index.search` (oversampling для фильтра по тегам).
3. Результаты обогащаются метаданными; далее возможен `merge`, `expand`, `compress`.

### Rebuild
*При смене модели или типа индекса* — пересчитываются все эмбеддинги, создаётся новый индекс без потери meta.

## 3. CapsuleStore и MetaStore

```mermaid
classDiagram
    class CapsuleStore {
        +model_name
        +dimension
        +index_factory
        +add_capsules()
        +query()
        +remove_capsules()
        +rebuild_index()
    }
    class MetaStore <|-- InMemoryMetaStore
    class MetaStore <|-- SQLiteMetaStore
```

* `CapsuleStore` отвечает за **векторы**
* `MetaStore` — за **любой JSON** (source, tags, custom поля)

## 4. Расширяемость

* Добавление нового `MetaStore` → реализовать CRUD + `_replace_all`.
* Новый тип индекса → использовать `faiss.index_factory` или написать обёртку.
* Подключение сторонней модели → реализовать `.encode()` и `.get_sentence_embedding_dimension()` (см. `TransformersEmbeddings`).

## 5. Коммуникация между процессами

| Механизм | Назначение |
|----------|------------|
| Stdout/CLI | batch jobs, cron |
| FastAPI    | online-режим, UI |
| Log JSONL  | audit, Prometheus scrape |

---

Подробнее смотрите в `IMPLEMENTATION.md` и комментариях к коду. 