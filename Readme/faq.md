# ЧаВО (FAQ)

### Q: Почему SIGLA, когда есть готовые облачные решения?
**A:** Полная приватность, оффлайн режим и контроль над индексом/весами.

### Q: Сколько данных выдержит Flat индекс?
**A:** Практика показывает ~100k капсул на CPU без деградации; далее рекомендуется IVF или IVF+PQ.

### Q: Можно ли использовать GPT-4 вместо локальной модели?
**A:** Да, реализуйте класс с тем же интерфейсом `encode()` и подключите через `CapsuleStore`.

### Q: Как обновить индексы после удаления капсул?
**A:** `sigla scripts dsl prune` или REST `/prune` – внутри вызывается `remove_capsules` с последующим `rebuild_index`.

### Q: Поддерживается ли Windows?
**A:** Да, тесты проходят на Python 3.11 + faiss-cpu.

### Q: Где найти примеры?
**A:** См. папку `examples/` (будет добавлена) и unit-тесты в `tests/`. 