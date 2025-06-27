#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы SIGLA
"""

import sys
import os
import json

def test_sigla():
    """Тест основных функций SIGLA"""
    print("🔥 Тестируем SIGLA...")
    
    try:
        # Импорт основных компонентов
        from sigla.core import CapsuleStore
        print("✅ Успешно импортировали CapsuleStore")
        
        # Создание индекса
        print("📦 Создаем индекс...")
        store = CapsuleStore(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print(f"✅ Индекс создан, размерность: {store.dimension}")
        
        # Загрузка капсул
        print("📁 Загружаем капсулы...")
        with open('sample_capsules.json', 'r', encoding='utf-8') as f:
            capsules = json.load(f)
        print(f"✅ Загружено {len(capsules)} капсул")
        
        # Добавление капсул в индекс
        print("🔍 Добавляем капсулы в индекс...")
        store.add_capsules(capsules)
        print("✅ Капсулы добавлены в индекс")
        
        # Сохранение индекса
        print("💾 Сохраняем индекс...")
        store.save("test_index")
        print("✅ Индекс сохранен")
        
        # Тестовый поиск
        print("🔍 Тестируем поиск...")
        results = store.query("машинное обучение", top_k=3)
        print(f"✅ Найдено {len(results)} результатов")
        
        for i, result in enumerate(results[:2]):
            print(f"   {i+1}. {result['text'][:80]}... (score: {result['score']:.3f})")
        
        # Тест DSL
        print("🧠 Тестируем DSL...")
        from sigla.dsl import INTENT, RETRIEVE, MERGE, INJECT
        from sigla.core import merge_capsules
        
        intent_vector = INTENT(store, "нейронные сети")
        retrieved = RETRIEVE(store, intent_vector, top_k=3)
        merged = MERGE(retrieved)
        injected = INJECT(merged)
        
        print("✅ DSL работает корректно")
        print(f"🎯 Пример инъекции:\n{injected[:200]}...")
        
        print("\n🎉 Все тесты прошли успешно!")
        print("💡 SIGLA готов к использованию!")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_sigla()
    sys.exit(0 if success else 1) 
