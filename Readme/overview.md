# Обзор проекта SIGLA

SIGLA (Semantic Information Graph & Lightweight Agents) — это модульная платформа для **семантического поиска**, **управления капсулами знаний** и **оркестрации локальных LLM-моделей**.

Главные особенности:

| Категория         | Возможности |
|-------------------|-------------|
| Поиск             | FAISS Flat → IVF100/IVF400, нормализация векторов, динамическое переобучение индекса, поиск по тегам |
| Инжест            | Автоматическое определение устройства (CPU/GPU/MPS), пакетная обработка, sliding-window суммаризация |
| Капсулы           | JSON-meta, auto-links (k-NN), формат `.capsulegraph` (веса модели + индекс) |
| DSL               | INTENT / RETRIEVE / MERGE / INJECT / EXPAND helpers |
| Сервер            | FastAPI, API-key auth, Prometheus-ready метрики |
| CLI               | `sigla scripts` (ingest, search, info, list-models, serve, dsl, convert) |
| Расширяемость     | Плагины meta-хранилищ (In-Memory, SQLite, DuckDB), GPTQ-4bit, Hydra config |
| CI/CD             | GitHub Actions (pytest + benchmark), Dockerfile (multi-stage) |

## Цели

1. **Локальная приватность** — всё работает оффлайн, без отправки запросов к внешним сервисам.
2. **Расширяемость** — чёткое разделение слоёв (CLI ↔ Core ↔ Meta-Store ↔ UI).
3. **Простота** — минимальные зависимости, понятный код, готовые Docker-образы.
4. **Производительность** — Faiss-GPU, quantize-friendly пайплайн, lazy-loading весов.

## Роадмап (выжимка)

| Спринт | Задачи |
|--------|--------|
| 1      | MetaStore → SQLite, Prometheus-метрики, тесты API |
| 2      | GPTQ 4-bit, IVF+PQ индексы, Docker slim |
| 3      | Hydra пресеты, >1M edges в графе, Grafana dashboards |

Полный план см. в `SIGLA_Plan.md`. 