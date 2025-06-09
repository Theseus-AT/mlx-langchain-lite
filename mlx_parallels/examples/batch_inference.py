#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

"""
MLX Parallels - Praktisches Verwendungsbeispiel
Zeigt die Integration mit mlx-langchain-lite und mlx-vector-db
"""

import asyncio
import time
from typing import List, Dict, Any
import logging

# MLX Parallels
from mlx_parallels.core.batch_processor import BatchProcessor
from mlx_parallels.core.config import get_high_throughput_config, get_fast_inference_config
from mlx_parallels.integration import EnhancedMLXLLMHandler, EnhancedRAGOrchestrator

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_basic_batch_processing():
    """Basis Batch-Processing Demo"""
    
    print("\n=== MLX Parallels - Basis Batch-Processing Demo ===")
    
    # Enhanced LLM Handler erstellen
    llm_handler = EnhancedMLXLLMHandler(
        model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
        performance_mode="fast"
    )
    
    # Initialisierung
    print("Modell wird geladen...")
    success = await llm_handler.initialize()
    if not success:
        print("❌ Modell konnte nicht geladen werden")
        return
    
    print("✅ Modell erfolgreich geladen")
    
    # Warmup
    print("Warmup wird durchgeführt...")
    await llm_handler.warmup()
    
    # Test-Prompts
    test_prompts = [
        "Erkläre was Machine Learning ist in einem Satz.",
        "Was sind die Vorteile von Apple Silicon?",
        "Wie funktioniert MLX Framework?",
        "Beschreibe die Unified Memory Architektur.",
        "Was ist der Unterschied zwischen CPU und GPU?"
    ]
    
    print(f"\nBatch-Generierung für {len(test_prompts)} Prompts...")
    
    # Batch-Generierung
    start_time = time.time()
    result = await llm_handler.batch_generate(
        prompts=test_prompts,
        max_tokens=50,
        temperature=0.7
    )
    
    total_time = time.time() - start_time
    
    print(f"\n✅ Batch-Generierung abgeschlossen:")
    print(f"   Zeit: {total_time:.2f}s")
    print(f"   Tokens: {result.tokens_generated}")
    print(f"   Geschwindigkeit: {result.tokens_per_second:.1f} tok/s")
    print(f"   Durchschnitt pro Prompt: {total_time/len(test_prompts):.2f}s")
    
    # Ergebnisse anzeigen
    print("\n--- Generierte Antworten ---")
    for i, (prompt, response) in enumerate(zip(test_prompts, result.outputs)):
        print(f"\n{i+1}. Prompt: {prompt}")
        print(f"   Antwort: {response.strip()}")
    
    # Performance-Statistiken
    stats = llm_handler.get_performance_stats()
    print(f"\n--- Performance-Statistiken ---")
    print(f"Total Requests: {stats['total_requests']}")
    print(f"Batch Requests: {stats['batch_requests']}")
    print(f"Durchschnittliche Latenz: {stats['avg_latency']:.3f}s")
    print(f"Memory Usage: {stats['memory_usage']}")
    
    await llm_handler.shutdown()


async def demo_rag_batch_processing():
    """RAG Batch-Processing Demo"""
    
    print("\n=== MLX Parallels - RAG Batch-Processing Demo ===")
    
    # Enhanced LLM Handler für RAG
    llm_handler = EnhancedMLXLLMHandler(
        model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
        performance_mode="balanced"
    )
    
    await llm_handler.initialize()
    await llm_handler.warmup()
    
    # Simulierte RAG-Daten
    queries = [
        "Was ist MLX Framework?",
        "Wie funktioniert Apple Silicon?",
        "Welche Vorteile hat Unified Memory?",
        "Was sind die Performance-Vorteile von MLX?"
    ]
    
    # Simulierte Context-Dokumente
    contexts = [
        [
            "MLX ist ein Array-Framework für Machine Learning auf Apple Silicon.",
            "MLX nutzt Unified Memory und Metal Performance Shaders.",
            "MLX bietet NumPy-ähnliche APIs für einfache Nutzung."
        ],
        [
            "Apple Silicon verwendet ARM-basierte Prozessoren.",
            "M1, M2 und M3 Chips bieten hohe Effizienz und Performance.",
            "Neural Engine ermöglicht beschleunigte ML-Operationen."
        ],
        [
            "Unified Memory ermöglicht geteilten Speicher zwischen CPU und GPU.",
            "Kein expliziter Datentransfer zwischen Speicherbereichen nötig.",
            "Reduziert Latenz und erhöht Effizienz."
        ],
        [
            "MLX ist speziell für Apple Silicon optimiert.",
            "Lazy Evaluation reduziert Speicherverbrauch.",
            "Metal Kernels bieten GPU-Beschleunigung."
        ]
    ]
    
    print(f"RAG Batch-Verarbeitung für {len(queries)} Queries...")
    
    # RAG Batch-Generierung
    start_time = time.time()
    rag_result = await llm_handler.rag_batch_generate(
        queries=queries,
        contexts=contexts,
        max_tokens=100,
        temperature=0.5
    )
    
    total_time = time.time() - start_time
    
    print(f"\n✅ RAG Batch-Generierung abgeschlossen:")
    print(f"   Zeit: {total_time:.2f}s")
    print(f"   Tokens: {rag_result.tokens_generated}")
    print(f"   Geschwindigkeit: {rag_result.tokens_per_second:.1f} tok/s")
    
    # RAG-Ergebnisse anzeigen
    print("\n--- RAG Antworten ---")
    for i, (query, response) in enumerate(zip(queries, rag_result.outputs)):
        print(f"\n{i+1}. Query: {query}")
        print(f"   RAG-Antwort: {response.strip()}")
        print(f"   Contexts verwendet: {len(contexts[i])}")
    
    await llm_handler.shutdown()


async def demo_performance_comparison():
    """Performance-Vergleich zwischen verschiedenen Modi"""
    
    print("\n=== MLX Parallels - Performance-Vergleich ===")
    
    test_prompts = [
        "Erkläre Künstliche Intelligenz.",
        "Was ist Deep Learning?",
        "Beschreibe Neural Networks.",
        "Wie funktioniert Backpropagation?",
        "Was sind Transformer-Modelle?"
    ] * 3  # 15 Prompts total
    
    performance_modes = ["fast", "balanced", "throughput"]
    results = {}
    
    for mode in performance_modes:
        print(f"\n--- Testing {mode.upper()} Modus ---")
        
        # Handler für aktuellen Modus
        handler = EnhancedMLXLLMHandler(
            model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
            performance_mode=mode
        )
        
        await handler.initialize()
        await handler.warmup()
        
        # Performance-Test
        start_time = time.time()
        result = await handler.batch_generate(
            prompts=test_prompts,
            max_tokens=30,
            temperature=0.1
        )
        total_time = time.time() - start_time
        
        results[mode] = {
            'total_time': total_time,
            'tokens_generated': result.tokens_generated,
            'tokens_per_second': result.tokens_per_second,
            'avg_time_per_prompt': total_time / len(test_prompts),
            'memory_usage': result.memory_usage
        }
        
        print(f"   Zeit: {total_time:.2f}s")
        print(f"   Tokens/s: {result.tokens_per_second:.1f}")
        print(f"   Memory: {result.memory_usage.get('system_available_gb', 'N/A'):.1f}GB verfügbar")
        
        await handler.shutdown()
    
    # Vergleichstabelle
    print("\n--- Performance-Vergleich ---")
    print("Modus       | Zeit (s) | Tok/s | Zeit/Prompt (s)")
    print("-" * 50)
    for mode, data in results.items():
        print(f"{mode:10} | {data['total_time']:7.2f} | {data['tokens_per_second']:5.0f} | {data['avg_time_per_prompt']:13.3f}")


async def demo_memory_optimization():
    """Memory-Optimierung Demo"""
    
    print("\n=== MLX Parallels - Memory-Optimierung Demo ===")
    
    handler = EnhancedMLXLLMHandler(
        model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
        performance_mode="balanced"
    )
    
    await handler.initialize()
    
    # Viele Test-Prompts für Memory-Test
    large_prompt_set = [
        f"Erkläre das Konzept {i}: Dies ist ein längerer Prompt um Memory-Verhalten zu testen." 
        for i in range(50)
    ]
    
    print(f"Memory-Test mit {len(large_prompt_set)} Prompts...")
    
    # Batch-Größe optimieren
    optimal_batch_size = await handler.optimize_batch_size(large_prompt_set[:5])
    print(f"Optimale Batch-Größe: {optimal_batch_size}")
    
    # Memory-Monitoring vor der Verarbeitung
    memory_before = handler.batch_processor.memory_manager.get_current_usage()
    print(f"Memory vor Verarbeitung: {memory_before['system_available_gb']:.1f}GB verfügbar")
    
    # Verarbeitung in optimalen Batches
    start_time = time.time()
    
    all_responses = []
    batch_size = optimal_batch_size
    
    for i in range(0, len(large_prompt_set), batch_size):
        batch = large_prompt_set[i:i + batch_size]
        
        batch_result = await handler.batch_generate(
            prompts=batch,
            max_tokens=20,
            temperature=0.1
        )
        
        all_responses.extend(batch_result.outputs)
        
        # Memory-Status loggen
        current_memory = handler.batch_processor.memory_manager.get_current_usage()
        print(f"Batch {i//batch_size + 1}: {current_memory['system_available_gb']:.1f}GB verfügbar")
        
        # Memory-Cleanup bei Bedarf
        handler.batch_processor.memory_manager.trigger_cleanup()
    
    total_time = time.time() - start_time
    
    # Memory-Status nach Verarbeitung
    memory_after = handler.batch_processor.memory_manager.get_current_usage()
    
    print(f"\n✅ Memory-Optimierung Test abgeschlossen:")
    print(f"   Prompts verarbeitet: {len(large_prompt_set)}")
    print(f"   Gesamtzeit: {total_time:.2f}s")
    print(f"   Memory vorher: {memory_before['system_available_gb']:.1f}GB")
    print(f"   Memory nachher: {memory_after['system_available_gb']:.1f}GB")
    print(f"   Memory-Differenz: {memory_after['system_available_gb'] - memory_before['system_available_gb']:.1f}GB")
    
    # Memory-Statistiken
    memory_stats = handler.batch_processor.memory_manager.get_memory_stats()
    print(f"\n--- Memory-Statistiken ---")
    print(f"Apple Silicon: {memory_stats['apple_silicon']}")
    print(f"Unified Memory: {memory_stats['unified_memory']}")
    print(f"MLX Peak Usage: {memory_stats['mlx_baseline_mb']:.1f}MB")
    
    await handler.shutdown()


async def demo_vector_db_integration():
    """Demo für VectorDB-Integration (Placeholder)"""
    
    print("\n=== MLX Parallels - VectorDB-Integration Demo ===")
    print("🚧 Placeholder für mlx-vector-db Integration")
    
    # Hier würde die echte Integration mit Ihrer mlx-vector-db stehen
    
    # Simulierte VectorDB-Batch-Operationen
    print("Simulierte Batch-Embedding für VectorDB...")
    
    texts_to_embed = [
        "MLX ist ein Framework für Apple Silicon",
        "Unified Memory bietet Performance-Vorteile",
        "Neural Engine beschleunigt ML-Workloads",
        "Metal Performance Shaders für GPU-Computing",
        "ARM-Prozessoren sind energieeffizient"
    ]
    
    # Hier würde BatchProcessor für Embeddings verwendet werden
    print(f"✅ {len(texts_to_embed)} Texte für Embedding vorbereitet")
    print("   Integration mit mlx-vector-db folgt...")
    
    # Simulierte Batch-Query
    queries = [
        "Was ist MLX?",
        "Wie funktioniert Unified Memory?",
        "Vorteile von Apple Silicon?"
    ]
    
    print(f"✅ {len(queries)} Queries für Batch-Vector-Search vorbereitet")
    print("   Batch-RAG-Pipeline bereit für Integration")


async def main():
    """Hauptfunktion - führt alle Demos aus"""
    
    print("🚀 MLX Parallels - Comprehensive Demo Suite")
    print("=" * 60)
    
    try:
        # Basis Demos
        await demo_basic_batch_processing()
        await demo_rag_batch_processing()
        
        # Performance Tests
        await demo_performance_comparison()
        await demo_memory_optimization()
        
        # Integration Demos
        await demo_vector_db_integration()
        
        print("\n🎉 Alle Demos erfolgreich abgeschlossen!")
        print("\nNächste Schritte:")
        print("1. Integration in mlx-langchain-lite")
        print("2. Verbindung mit mlx-vector-db")
        print("3. Production-ready Konfiguration")
        
    except Exception as e:
        logger.error(f"Demo fehlgeschlagen: {e}")
        raise


if __name__ == "__main__":
    # Performance Monitoring
    import psutil
    
    print(f"System Info:")
    print(f"  CPU: {psutil.cpu_count()} cores")
    print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    print(f"  Platform: {psutil.platform}")
    
    # Hauptdemo ausführen
    asyncio.run(main())