import os
import json
from dotenv import load_dotenv
from rag_system import rag_system
from typing import Dict, Any

# === Загрузка переменных среды ===
load_dotenv()

def print_search_results(result: Dict[str, Any]):
    """Pretty print search results"""
    print("\n" + "="*60)
    print("🔍 РЕЗУЛЬТАТЫ ПОИСКА")
    print("="*60)
    
    print(f"\n📝 ОТВЕТ:")
    print("-" * 40)
    print(result["answer"])
    
    print(f"\n📚 ИСТОЧНИКИ ({len(result['sources'])}):")
    print("-" * 40)
    for i, source in enumerate(result["sources"], 1):
        print(f"{i}. {source}")
    
    print(f"\n🔍 ДЕТАЛИ ПОИСКА:")
    print("-" * 40)
    print(f"• Найдено результатов: {result['results_count']}")
    print(f"• Длина контекста: {result['context_length']} символов")
    
    if result["search_results"]:
        print(f"\n📄 ТОП РЕЗУЛЬТАТЫ:")
        print("-" * 40)
        for i, search_result in enumerate(result["search_results"][:3], 1):
            print(f"\n{i}. {search_result['source']} (score: {search_result['score']:.4f})")
            print(f"   {search_result['text']}")

def interactive_query():
    """Interactive query interface"""
    print("🤖 Enhanced Legal RAG System")
    print("=" * 50)
    print("Доступные команды:")
    print("  /help - показать справку")
    print("  /stats - показать статистику системы")
    print("  /history - показать историю разговора")
    print("  /clear - очистить историю")
    print("  /quit - выйти")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n💬 Введите юридический запрос (или команду): ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.startswith("/"):
                handle_command(user_input)
                continue
            
            # Perform search
            print("\n🔄 Обрабатываю запрос...")
            
            # Try hybrid search first
            result = rag_system.query(
                user_query=user_input,
                use_hybrid_search=True,
                use_reranking=True
            )
            
            print_search_results(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}")

def handle_command(command: str):
    """Handle special commands"""
    cmd = command.lower()
    
    if cmd == "/help":
        print("\n📖 СПРАВКА:")
        print("-" * 30)
        print("• Введите любой юридический вопрос")
        print("• Система найдет релевантные документы")
        print("• Ответ будет основан на найденном контексте")
        print("• История разговора сохраняется")
        
    elif cmd == "/stats":
        print("\n📊 СТАТИСТИКА СИСТЕМЫ:")
        print("-" * 30)
        stats = rag_system.get_system_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"• {key}:")
                for sub_key, sub_value in value.items():
                    print(f"  - {sub_key}: {sub_value}")
            else:
                print(f"• {key}: {value}")
                
    elif cmd == "/history":
        print("\n📜 ИСТОРИЯ РАЗГОВОРА:")
        print("-" * 30)
        history = rag_system.get_conversation_history()
        if not history:
            print("История пуста")
        else:
            for i, turn in enumerate(history[-5:], 1):  # Show last 5 turns
                print(f"\n{i}. Вопрос: {turn['user_query']}")
                print(f"   Ответ: {turn['response'][:100]}...")
                print(f"   Источники: {', '.join(turn['sources'][:2])}")
                
    elif cmd == "/clear":
        rag_system.clear_conversation_history()
        print("\n✅ История разговора очищена")
        
    elif cmd == "/quit":
        print("\n👋 До свидания!")
        exit(0)
        
    else:
        print(f"\n❌ Неизвестная команда: {command}")
        print("Используйте /help для справки")

def single_query(query: str):
    """Perform a single query"""
    print(f"\n🔍 Поиск по запросу: {query}")
    print("=" * 50)
    
    result = rag_system.query(
        user_query=query,
        use_hybrid_search=True,
        use_reranking=True
    )
    
    print_search_results(result)
    
    return result

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Command line query
        query = " ".join(sys.argv[1:])
        single_query(query)
    else:
        # Interactive mode
        interactive_query()
