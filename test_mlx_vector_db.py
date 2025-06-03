#!/usr/bin/env python3
"""
Spezifischer Test für Theseus-AT/mlx-vector-db Integration
Basiert auf der offiziellen API Documentation
"""

import sys
import json
import time
import requests
import numpy as np
from pathlib import Path
from typing import Optional

# API Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "your-secure-key"  # Aus .env file
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

def check_mlx_vector_db_installation():
    """Prüft ob MLXVectorDB installiert ist"""
    print("🔍 Checking MLXVectorDB Installation...")
    
    try:
        # Versuche mlx-vector-db zu importieren
        import mlx_vector_db
        print("✅ mlx-vector-db package imported successfully")
        return True
    except ImportError:
        print("❌ mlx-vector-db package not found")
        print("   Install with: pip install git+https://github.com/Theseus-AT/mlx-vector-db.git")
        return False

def check_server_health():
    """Prüft ob MLXVectorDB Server läuft"""
    print("\n🌐 Checking MLXVectorDB Server Health...")
    
    try:
        # Health Check Endpoint
        response = requests.get(f"{BASE_URL}/monitoring/health", timeout=5)
        
        if response.status_code == 200:
            health_data = response.json()
            print("✅ MLXVectorDB Server is running")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            print(f"   MLX Version: {health_data.get('mlx_version', 'unknown')}")
            print(f"   Server Time: {health_data.get('timestamp', 'unknown')}")
            return True
        else:
            print(f"⚠️  Server responded with status: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to MLXVectorDB Server")
        print(f"   Expected URL: {BASE_URL}")
        print("   Start server with: python main.py")
        return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def check_mlx_performance():
    """Prüft MLX Performance Status"""
    print("\n⚡ Checking MLX Performance...")
    
    try:
        response = requests.get(f"{BASE_URL}/performance/health", headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            perf_data = response.json()
            print("✅ MLX Performance Status:")
            print(f"   MLX Framework: {perf_data.get('mlx_version', 'unknown')}")
            print(f"   Device: {perf_data.get('device', 'unknown')}")
            print(f"   Memory Available: {perf_data.get('memory_gb', 'unknown')} GB")
            print(f"   Platform: {perf_data.get('platform', 'unknown')}")
            return True
        else:
            print(f"⚠️  Performance check failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Performance check failed: {e}")
        return False

def test_store_creation():
    """Testet Store Creation"""
    print("\n🏗️ Testing Store Creation...")
    
    try:
        create_payload = {
            "user_id": "test_user_mlx_langchain",
            "model_id": "test_model_integration"
        }
        
        response = requests.post(
            f"{BASE_URL}/admin/create_store", 
            json=create_payload, 
            headers=HEADERS,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Store created successfully")
            print(f"   Store Path: {result.get('store_path', 'unknown')}")
            print(f"   User ID: {result.get('user_id', 'unknown')}")
            print(f"   Model ID: {result.get('model_id', 'unknown')}")
            return True
        elif response.status_code == 409:
            print("✅ Store already exists (conflict is expected)")
            return True
        else:
            print(f"❌ Store creation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Store creation test failed: {e}")
        return False

def test_vector_operations():
    """Testet Vector Add und Query Operations"""
    print("\n🧮 Testing Vector Operations...")
    
    try:
        # 1. Test Vector Addition
        print("   📥 Testing vector addition...")
        
        # Erstelle Test-Vektoren (384 dimensional, wie in der Doku)
        vectors = np.random.rand(10, 384).astype(np.float32)
        metadata = [
            {
                "id": f"test_doc_{i}",
                "source": "mlx_langchain_test",
                "content": f"Test document content {i}",
                "timestamp": time.time()
            }
            for i in range(10)
        ]
        
        add_payload = {
            "user_id": "test_user_mlx_langchain",
            "model_id": "test_model_integration",
            "vectors": vectors.tolist(),
            "metadata": metadata
        }
        
        response = requests.post(
            f"{BASE_URL}/vectors/add",
            json=add_payload,
            headers=HEADERS,
            timeout=15
        )
        
        if response.status_code == 200:
            add_result = response.json()
            print(f"   ✅ Added {add_result.get('count', 0)} vectors")
            print(f"   📊 Total vectors in store: {add_result.get('total_count', 'unknown')}")
        else:
            print(f"   ❌ Vector addition failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        # 2. Test Vector Query
        print("   🔍 Testing vector query...")
        
        query_vector = vectors[0].tolist()  # Use first vector as query
        query_payload = {
            "user_id": "test_user_mlx_langchain",
            "model_id": "test_model_integration",
            "query": query_vector,
            "k": 5
        }
        
        response = requests.post(
            f"{BASE_URL}/vectors/query",
            json=query_payload,
            headers=HEADERS,
            timeout=10
        )
        
        if response.status_code == 200:
            query_result = response.json()
            results = query_result.get('results', [])
            print(f"   ✅ Query returned {len(results)} results")
            
            if results:
                print(f"   🎯 Best match score: {results[0].get('score', 'unknown')}")
                print(f"   📄 Best match ID: {results[0].get('metadata', {}).get('id', 'unknown')}")
            
            return True
        else:
            print(f"   ❌ Vector query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Vector operations test failed: {e}")
        return False

def test_batch_operations():
    """Testet Batch Query Operations"""
    print("\n📦 Testing Batch Operations...")
    
    try:
        # Erstelle mehrere Query-Vektoren
        query_vectors = np.random.rand(3, 384).astype(np.float32)
        
        batch_payload = {
            "user_id": "test_user_mlx_langchain",
            "model_id": "test_model_integration",
            "queries": query_vectors.tolist(),
            "k": 3
        }
        
        response = requests.post(
            f"{BASE_URL}/vectors/batch_query",
            json=batch_payload,
            headers=HEADERS,
            timeout=15
        )
        
        if response.status_code == 200:
            batch_result = response.json()
            results = batch_result.get('results', [])
            print(f"✅ Batch query processed {len(results)} queries")
            
            for i, result in enumerate(results):
                matches = result.get('results', [])
                print(f"   Query {i+1}: {len(matches)} matches")
            
            return True
        else:
            print(f"❌ Batch query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch operations test failed: {e}")
        return False

def test_performance_benchmark():
    """Testet Performance Benchmark"""
    print("\n📊 Testing Performance Benchmark...")
    
    try:
        benchmark_payload = {
            "vector_count": 100,
            "dimension": 384,
            "query_count": 10
        }
        
        response = requests.post(
            f"{BASE_URL}/performance/benchmark",
            json=benchmark_payload,
            headers=HEADERS,
            timeout=30
        )
        
        if response.status_code == 200:
            benchmark_result = response.json()
            print("✅ Performance Benchmark Results:")
            print(f"   Vector Addition Rate: {benchmark_result.get('add_rate', 'unknown')} vectors/sec")
            print(f"   Query Performance: {benchmark_result.get('query_qps', 'unknown')} QPS")
            print(f"   Average Latency: {benchmark_result.get('avg_latency', 'unknown')} ms")
            print(f"   MLX Speedup: {benchmark_result.get('speedup', 'unknown')}x")
            return True
        else:
            print(f"⚠️  Benchmark failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"⚠️  Benchmark test failed: {e}")
        print("   This is optional - basic functionality should still work")
        return False

def test_store_statistics():
    """Testet Store Statistics"""
    print("\n📈 Testing Store Statistics...")
    
    try:
        params = {
            "user_id": "test_user_mlx_langchain",
            "model_id": "test_model_integration"
        }
        
        response = requests.get(
            f"{BASE_URL}/vectors/count",
            params=params,
            headers=HEADERS,
            timeout=5
        )
        
        if response.status_code == 200:
            stats = response.json()
            print("✅ Store Statistics:")
            print(f"   Total Vectors: {stats.get('count', 'unknown')}")
            print(f"   Store Size: {stats.get('size_mb', 'unknown')} MB")
            print(f"   Last Updated: {stats.get('last_updated', 'unknown')}")
            return True
        else:
            print(f"⚠️  Statistics failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")
        return False

def print_integration_help():
    """Druckt Integration Hilfe für mlx-langchain-lite"""
    print("\n" + "="*60)
    print("🔗 MLXVectorDB Integration für mlx-langchain-lite")
    print("="*60)
    print()
    print("Konfiguration in mlx_components/vector_store.py:")
    print()
    print("```python")
    print("# In deiner VectorStoreConfig:")
    print("class VectorStoreConfig:")
    print(f"    base_url: str = '{BASE_URL}'")
    print(f"    api_key: str = '{API_KEY}'")
    print("    collection_name: str = 'mlx_langchain_vectors'")
    print("    dimension: int = 384")
    print("    timeout: int = 30")
    print("```")
    print()
    print("Environment Variables (.env):")
    print(f"VECTOR_DB_URL={BASE_URL}")
    print(f"VECTOR_DB_API_KEY={API_KEY}")
    print()
    print("MLXVectorDB Server starten:")
    print("cd /pfad/zu/mlx-vector-db")
    print("python main.py")
    print()

def main():
    """Haupt-Test für MLXVectorDB Integration"""
    print("🎯 MLXVectorDB Integration Test für mlx-langchain-lite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 8
    
    # Test Suite
    tests = [
        ("Installation", check_mlx_vector_db_installation),
        ("Server Health", check_server_health),
        ("MLX Performance", check_mlx_performance),
        ("Store Creation", test_store_creation),
        ("Vector Operations", test_vector_operations),
        ("Batch Operations", test_batch_operations),
        ("Performance Benchmark", test_performance_benchmark),
        ("Store Statistics", test_store_statistics),
    ]
    
    for test_name, test_func in tests:
        try:
            if test_func():
                tests_passed += 1
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed >= 6:
        print("🎉 MLXVectorDB Integration erfolgreich!")
        print("   Bereit für mlx-langchain-lite Integration!")
    elif tests_passed >= 3:
        print("⚠️  Grundfunktionalität OK, aber einige erweiterte Features fehlen")
    else:
        print("❌ Integration fehlgeschlagen")
        print("   Überprüfe Server und Konfiguration")
    
    print_integration_help()

if __name__ == "__main__":
    main()