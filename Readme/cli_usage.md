# Использование CLI (`sigla scripts`)

`sigla scripts` — единая точка входа к большинству функций SIGLA. Ниже приведены часто используемые под-команды.

> Подсказка: для справки по любой команде выполните `sigla scripts <command> -h`.

## 1. ingest

Ингестирует файлы/директории в CapsuleStore.

```bash
sigla scripts ingest <input> --output <store_name> [options]
```

| Параметр           | Описание                                   | По умолчанию |
|--------------------|--------------------------------------------|--------------|
| `<input>`          | Файл (`.txt`, `.md`, `.json`) или директория| — |
| `-o, --output`     | Имя выходного хранилища (без расширений)   | `capsules` |
| `-m, --model`      | HF модель для эмбеддингов                  | `all-MiniLM-L6-v2` |
| `--local-model`    | Путь к локальной модели (`config.json` и веса) | — |
| `--auto-model`     | Автоматически выбрать первую локальную модель | `false` |
| `--device`         | `auto` / `cpu` / `cuda` / `mps`            | `auto` |

Пример:
```bash
sigla scripts ingest docs/ --output wiki --device cuda --auto-model
```

## 2. search

```bash
sigla scripts search "<query>" --store <store_name> -k 5 [--tags tag1 tag2]
```

Результатом будет список топ-K капсул с метаданными и score.

## 3. info

Возвращает сводку о CapsuleStore (модель, размерность, количество капсул, топ тегов).

```bash
sigla scripts info --store wiki
```

## 4. list-models

Сканирует папку `./models` и выводит найденные локальные модели.

```bash
sigla scripts list-models
```

## 5. serve

Запускает FastAPI сервер поверх выбранного магазина.

```bash
sigla scripts serve --store wiki --port 8000 --device cpu
```

## 6. convert

Пакует HF-модель в `.capsulegraph` архив (квантизация int8/fp16).

```bash
sigla scripts convert /path/to/model out.cg --bits 8 --device cuda
```

## 7. run-cg

Генерирует текст при помощи заранее сконвертированного `.capsulegraph`.

```bash
sigla scripts run-cg out.cg --prompt "Hello" --max-tokens 128
```

## 8. dsl

Выполняет одну из DSL-операций (INTENT, RETRIEVE, MERGE, INJECT, EXPAND) поверх хранилища.

```bash
sigla scripts dsl retrieve "quantum computing" --store wiki -k 3
``` 