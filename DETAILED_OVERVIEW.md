# SIGLA – Подробный обзор проекта

## 1. Назначение
SIGLA (Semantic Information Graph with Language Agents) – лёгкая платформа для создания и эксплуатации **семантического графа знаний** на базе локальных моделей.

* Каждая *капсула* – небольшой текст с метаданными (теги, ссылки).
* Система кодирует капсулы в векторы, индексирует их через FAISS и предоставляет быстрый поиск по семантическому сходству.
* Работает полностью локально (конфиденциальность данных), не требует внешних API.

---

## 2. Ключевая функциональность
| Компонент | Описание |
|-----------|----------|
| **TransformersEmbeddings** | Универсальная обёртка для любых HF-моделей. Авто-выбор устройства (CUDA › MPS › CPU). |
| **CapsuleStore** | Добавление, поиск, удаление капсул; сохранение/загрузка *.index & *.json; перестройка индекса. |
| **DSL-слой** | `INTENT`, `RETRIEVE`, `MERGE`, `INJECT`, `EXPAND` – высокоуровневые операции. |
| **Graph utils** | `expand_with_links`, `random_walk_links` – обход графа капсул. |
| **CLI** | Команды `ingest`, `search`, `info`, `list-models`, `serve`, `dsl`. |
| **FastAPI-сервер** | REST-эндпоинты для поиска, агрегации, графовых операций и администрирования. |
| **Авто-детект моделей** | Находит локальные HF-модели, выбирает «самую большую» по объёму. |
| **Логирование** | `sigla/log.py` – JSON-лог событий CLI и API. |

---

## 3. Возможные применения
* Персональная база знаний (альтернатива Obsidian + GPT-плагинам).
* Локальный FAQ-движок / справочная система.
* Сжатие контекста для LLM-агента (память).
* Семантический поиск по кодовой базе, статьям, стенограммам.
* Образовательные графы: конспекты, карточки терминов, связи.

---

## 4. Сильные стороны
* Полностью офлайн; конфиденциальность.
* Поддержка **любых** HF-моделей (cpu/gpu/mps).
* Простой формат хранения (FAISS + JSON).
* Универсальный доступ: Python-API, CLI, REST.
* Минимальные зависимости, graceful-fallback при отсутствии torch/transformers.

---

## 5. Слабые места / нелогичности
* Отсутствуют unit-тесты и CI.
* Авто-выбор «самой тяжёлой» модели ➜ иногда нужна более быстрая.
* `compress_capsules` обрезает текст >4 к симв.
* Связи (`links`) заполняются вручную – нет авто-линковки.
* REST-API без аутентификации и rate-limit.
* Нет полноценного веб-UI.
* Дублирование списков зависимостей (requirements vs setup.py).

---

## 6. Потенциальные улучшения
1. **Тесты + CI** (pytest + GitHub Actions).
2. Dockerfile / docker-compose для мгновенного развёртывания.
3. Автоматическая линковка капсул (k-NN) при ingest.
4. Визуальный Web-UI (Streamlit/Svelte).
5. Расширенный DSL-язык + парсер-интерпретатор.
6. Метрики и мониторинг (Prometheus).
7. Версионирование схемы индекса и миграции.
8. Индексы HNSW / IVF-PQ для больших корпусов.

---

## 7. Производительность (ориентиры)
| Операция | MiniLM/CPU | MiniLM/GPU |
|----------|-----------|-----------|
| Кодирование | ~500 текстов/с | >1500 текстов/с |
| Поиск (Flat) | <1 мс при 10k векторов | <1 мс |
| Загрузка индекса | секунды | секунды |

---

*Документ обновлён: 2025-06-25.* 