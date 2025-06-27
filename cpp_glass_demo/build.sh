#!/usr/bin/env bash
# Compile demo with Emscripten
em++ main.cpp -std=c++17 -s USE_GLFW=3 -s FULL_ES2=1 -s MINIFY_HTML=0 -O2 -o glass.html
