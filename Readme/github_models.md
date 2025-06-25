# Интеграция с GitHub Models

> Эта страница описывает, как использовать службу **GitHub Models** (предпросмотр) для вызова больших языковых моделей (LLM) из-под кода SIGLA. Мы будем использовать модель `mistral-ai/Codestral-2501`, но процесс одинаков для других доступных моделей.

Док официально описан в [About GitHub Models](https://docs.github.com/ru/github-models/about-github-models). Ниже приведён краткий конспект и практическое руководство.

---

## 1. Получение токена

1. Перейдите в **Settings → Developer settings → Personal access tokens**.
2. Создайте **Fine-grained token**.
3. Выдайте разрешение **models:read** (без него будет *401 Unauthorized*).
4. Скопируйте токен `ghp_…` и сохраните в переменной окружения:
   ```powershell
   $Env:GITHUB_TOKEN="<your-token>"
   ```
   > Токен будет отправляться на сервис Microsoft (хостинг моделей) — учтите корпоративные требования к безопасности.

---

## 2. Установка зависимостей

```bash
pip install mistralai>=1.0.0  # Python ≥3.9
```

В `setup.py` уже предусмотрен extra-тэг `llm`, добавьте `mistralai` при необходимости.

---

## 3. Базовый вызов чата

```python
import os
from mistralai import Mistral, UserMessage, SystemMessage

endpoint = "https://models.github.ai/inference"
model   = "mistral-ai/Codestral-2501"

token   = os.environ["GITHUB_TOKEN"]
client  = Mistral(api_key=token, server_url=endpoint)

response = client.chat.complete(
    model=model,
    messages=[
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What is the capital of France?"),
    ],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)
```

---

## 4. Потоковая передача (stream)

```python
for chunk in client.chat.stream(
    model=model,
    messages=[UserMessage(content="Сгенерируй Limerick про SIGLA")],
):
    if chunk.data.choices:
        print(chunk.data.choices[0].delta.content or "", end="")
```

---

## 5. Интеграция в SIGLA

### 5.1 Расширение модуля `sigla.llm`

Добавьте класс-обёртку:
```python
class GitHubLLM:
    def __init__(self, model: str = "mistral-ai/Codestral-2501", temp: float = 0.7):
        from mistralai import Mistral, UserMessage, SystemMessage
        import os
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise RuntimeError("GITHUB_TOKEN not set")
        self._client = Mistral(api_key=token, server_url="https://models.github.ai/inference")
        self._model  = model
        self._temp   = temp

    def chat(self, prompt: str, system: str = "You are a helpful assistant.") -> str:
        from mistralai import UserMessage, SystemMessage
        resp = self._client.chat.complete(
            model=self._model,
            messages=[SystemMessage(content=system), UserMessage(content=prompt)],
            temperature=self._temp,
        )
        return resp.choices[0].message.content
```

Теперь можно использовать в скриптах, например, для summarization:
```python
from sigla.llm import GitHubLLM
...
llm = GitHubLLM()
summary = llm.chat(merge_capsules(caps))
```

### 5.2 FastAPI endpoint `/llm`

В `sigla/server.py`:
```python
@app.post("/llm")
def llm_endpoint(prompt: str, temperature: float = 0.7):
    from sigla.llm import GitHubLLM
    llm = GitHubLLM(temp=temperature)
    return {"answer": llm.chat(prompt)}
```

---

## 6. Ограничения скорости и продакшн-режим

* **Игровая площадка** и вызовы через PAT ограничены по RPS и количеству токенов.
* Для масштабирования используйте Azure AI Foundry: меняется только метод аутентификации.

---

## 7. Хранение промптов

GitHub Models поддерживает `.prompt.yml` в репозитории. Рекомендуется положить файлы в `prompts/` и версионировать их вместе с кодом.

---

### Полезные ссылки
* Док: [About GitHub Models](https://docs.github.com/ru/github-models/about-github-models) 