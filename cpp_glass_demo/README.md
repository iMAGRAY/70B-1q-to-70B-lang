# Glass Shader Demo

Пример C++ программы с анимированным шейдером "Liquid Glass". 
Скомпилировать можно через [Emscripten](https://emscripten.org/),
что позволит запустить демонстрацию прямо в браузере.

## Сборка
1. Установите Emscripten и активируйте окружение (`source /path/to/emsdk/emsdk_env.sh`).
2. В каталоге `cpp_glass_demo` выполните:
   ```bash
   ./build.sh
   ```
   Появятся файлы `glass.html`, `glass.js` и `glass.wasm`.
3. Для запуска используйте скрипт `./run.sh`, который поднимет простой HTTP‑сервер на порту 8080.
4. Откройте `http://localhost:8080/glass.html` в браузере.

Шейдер использует время для создания полупрозрачных волн, имитируя жидкое стекло.
