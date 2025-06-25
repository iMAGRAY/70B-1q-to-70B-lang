# Быстрый старт

## Установка

```bash
# Клонируем репозиторий
git clone https://github.com/your-org/sigla.git
cd sigla

# Создаём виртуальное окружение
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Устанавливаем пакет (+ ключевые зависимости)
pip install -e .[core]
```

> ⚠️  Для GPU-ускорения: `pip install faiss-gpu torch --extra-index-url https://download.pytorch.org/whl/cu118`.

---

## Первые шаги

1. **Ингест документов**
   ```bash
   sigla scripts ingest docs/ --output mycaps
   ```
2. **Поиск**
   ```bash
   sigla scripts search "what is vector search" --store mycaps -k 5
   ```
3. **Запуск FastAPI сервера**
   ```bash
   sigla scripts serve --store mycaps --port 8080
   ```
   Затем: `curl http://localhost:8080/search?query=hello`.

---

## Docker (альтернатива)

```bash
docker build -t sigla:latest .
# Запуск сервера внутри контейнера
docker run -p 8000:8000 sigla:latest sigla scripts serve --device cpu
``` 