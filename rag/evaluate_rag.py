import time
import json
from typing import List, Dict, Any
from rag_system import rag_system

# Sample test queries for legal domain
TEST_QUERIES = [
    "Какие права имеет работник при увольнении?",
    "Как оформить договор купли-продажи недвижимости?",
    "Какие документы нужны для регистрации ИП?",
    "Как подать иск в суд?",
    "Какие налоги платит физическое лицо?",
    "Как защитить авторские права?",
    "Какие условия трудового договора обязательны?",
    "Как оформить наследство?",
    "Какие права у потребителя при покупке товара?",
    "Как подать жалобу на действия чиновников?"
]

def evaluate_query(query: str, use_hybrid: bool = True, use_rerank: bool = True) -> Dict[str, Any]:
    """Evaluate a single query"""
    start_time = time.time()
    
    try:
        result = rag_system.query(
            user_query=query,
            use_hybrid_search=use_hybrid,
            use_reranking=use_rerank
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        return {
            "query": query,
            "success": True,
            "response_time": response_time,
            "answer_length": len(result["answer"]),
            "sources_count": len(result["sources"]),
            "results_count": result["results_count"],
            "context_length": result["context_length"],
            "has_answer": len(result["answer"]) > 50,
            "has_sources": len(result["sources"]) > 0
        }
        
    except Exception as e:
        end_time = time.time()
        return {
            "query": query,
            "success": False,
            "response_time": end_time - start_time,
            "error": str(e)
        }

def run_evaluation():
    """Run comprehensive evaluation"""
    print("🧪 EVALUATION OF ENHANCED RAG SYSTEM")
    print("=" * 60)
    
    # Test different configurations
    configurations = [
        ("Simple Search", False, False),
        ("Dense Search", False, True),
        ("Hybrid Search", True, True)
    ]
    
    results = {}
    
    for config_name, use_hybrid, use_rerank in configurations:
        print(f"\n🔍 Testing: {config_name}")
        print("-" * 40)
        
        config_results = []
        total_time = 0
        success_count = 0
        
        for i, query in enumerate(TEST_QUERIES, 1):
            print(f"  {i}/{len(TEST_QUERIES)}: {query[:50]}...")
            
            result = evaluate_query(query, use_hybrid, use_rerank)
            config_results.append(result)
            
            if result["success"]:
                success_count += 1
                total_time += result["response_time"]
        
        # Calculate metrics
        avg_time = total_time / success_count if success_count > 0 else 0
        success_rate = success_count / len(TEST_QUERIES)
        
        # Quality metrics
        avg_answer_length = sum(r["answer_length"] for r in config_results if r["success"]) / success_count if success_count > 0 else 0
        avg_sources = sum(r["sources_count"] for r in config_results if r["success"]) / success_count if success_count > 0 else 0
        
        results[config_name] = {
            "success_rate": success_rate,
            "avg_response_time": avg_time,
            "avg_answer_length": avg_answer_length,
            "avg_sources": avg_sources,
            "detailed_results": config_results
        }
        
        print(f"  ✅ Success Rate: {success_rate:.2%}")
        print(f"  ⏱️  Avg Response Time: {avg_time:.2f}s")
        print(f"  📝 Avg Answer Length: {avg_answer_length:.0f} chars")
        print(f"  📚 Avg Sources: {avg_sources:.1f}")
    
    # Print comparison
    print("\n📊 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<20} {'Success':<10} {'Time(s)':<10} {'Length':<10} {'Sources':<10}")
    print("-" * 60)
    
    for config_name, metrics in results.items():
        print(f"{config_name:<20} {metrics['success_rate']:<10.1%} {metrics['avg_response_time']:<10.2f} "
              f"{metrics['avg_answer_length']:<10.0f} {metrics['avg_sources']:<10.1f}")
    
    # Save detailed results
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Detailed results saved to: evaluation_results.json")
    
    return results

def test_conversation_memory():
    """Test conversation memory functionality"""
    print("\n💬 TESTING CONVERSATION MEMORY")
    print("=" * 40)
    
    # Clear history first
    rag_system.clear_conversation_history()
    
    # Simulate conversation
    conversation = [
        "Что такое трудовой договор?",
        "Какие обязательные условия он должен содержать?",
        "Можно ли расторгнуть договор досрочно?"
    ]
    
    for i, query in enumerate(conversation, 1):
        print(f"\n{i}. Вопрос: {query}")
        result = rag_system.query(query)
        print(f"   Ответ: {result['answer'][:100]}...")
        print(f"   Источники: {len(result['sources'])}")
    
    # Check history
    history = rag_system.get_conversation_history()
    print(f"\n📜 История содержит {len(history)} записей")
    
    return history

def test_system_stats():
    """Test system statistics"""
    print("\n📊 SYSTEM STATISTICS")
    print("=" * 30)
    
    stats = rag_system.get_system_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    # Run evaluation
    evaluation_results = run_evaluation()
    
    # Test conversation memory
    conversation_history = test_conversation_memory()
    
    # Test system stats
    test_system_stats()
    
    print("\n✅ Evaluation completed!") 