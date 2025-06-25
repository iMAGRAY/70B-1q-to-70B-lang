# Гайд разработчика

## Структура репозитория

```
sigla/            # исходный код пакета
  core.py         # CapsuleStore + Embeddings
  scripts.py      # CLI
  server.py       # FastAPI
  dsl.py          # мини DSL
  meta/           # backends для метаданных
  tests/          # pytest
Readme/           # документация (этот каталог)
Dockerfile        # multi-stage build
setup.py          # install / editable mode
```

## Настройка окружения

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pre-commit install  # black+flake8+mypy
```

## Запуск тестов и линтеров

```bash
pytest -q                 # unit-tests
black --check sigla/      # форматирование
flake8 sigla/             # стиль
mypy sigla/               # статический типинг
```

## Советы по код-стайлу

* Black 120 cols, isort внутри black.
* type hints обязательны для всех public функций.
* Один класс/функция — один файл до 400 LoC; дробите дальше.

## Расширение MetaStore

1. Создайте класс, наследующийся от `sigla.meta.base.MetaStore`.
2. Реализуйте минимум `add_many` и `all`.
3. Зарегистрируйте в `sigla.meta.__init__` (factory).

## Диагностика производительности

* `pytest -q --benchmark-only` (см. `pytest-benchmark`).
* Используйте `faiss.index_factory` для быстрой смены индекса.
* Для GPU активируйте `faiss-gpu` + `torch.cuda`.

## Выпуск релиза

```bash
git tag v0.X.Y
python -m build
pip upload dist/*  # twine или pypi-publish action
``` 