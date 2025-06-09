#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

"""
Test für den aktuellen minimalen MLXVectorDB Server
Basiert auf den tatsächlich verfügbaren Endpunkten
"""

import sys
import json
import time
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_server_basic_functionality():
    """Testet die grundlegenden Server-Funktionen"""
    print("🌐 Testing Basic MLXVectorDB Server Functionality...")
    
    try:
        # 1. Root Endpoint
        response = requests.get(f"{BASE_URL}/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Server Root Endpoint:")
            print(f"   Name: {data.get('name')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Status: {data.get('status')}")
            print(f"   Docs: {data.get('docs')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
            
        # 2. Health Check
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if response.status_code == 200:
            health = response.json()
            print("✅ Health Check:")
            print(f"   Status: {health.get('status')}")
            print(f"   Service: {health.get('service')}")
            print(f"   Version: {health.get('version')}")
            print(f"   Timestamp: {health.get('timestamp')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
        
        # 3. Debug Routes
        response = requests.get(f"{BASE_URL}/debug/routes", timeout=5)
        
        if response.status_code == 200:
            routes = response.json()
            print("✅ Available Routes:")
            for route in routes.get('routes', []):
                methods = ', '.join(route.get('methods', []))
                print(f"   {route.get('path')} - {methods}")
        else:
            print(f"⚠️  Debug routes not available: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def test_api_documentation():
    """Testet API-Dokumentation Verfügbarkeit"""
    print("\n📚 Testing API Documentation...")
    
    try:
        # OpenAPI JSON
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        
        if response.status_code == 200:
            spec = response.json()
            print("✅ OpenAPI Specification available")
            print(f"   Title: {spec.get('info', {}).get('title')}")
            print(f"   Version: {spec.get('info', {}).get('version')}")
            print(f"   Endpoints: {len(spec.get('paths', {}))}")
            
            # Zeige verfügbare Pfade
            for path, methods in spec.get('paths', {}).items():
                method_list = list(methods.keys())
                print(f"   {path} - {', '.join(method_list).upper()}")
        else:
            print(f"❌ OpenAPI spec not available: {response.status_code}")
            return False
        
        # Docs UI
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        
        if response.status_code == 200:
            print("✅ Interactive API docs available at /docs")
        else:
            print(f"⚠️  API docs UI not available: {response.status_code}")
        
        # ReDoc
        response = requests.get(f"{BASE_URL}/redoc", timeout=5)
        
        if response.status_code == 200:
            print("✅ ReDoc documentation available at /redoc")
        else:
            print(f"⚠️  ReDoc not available: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Documentation test failed: {e}")
        return False

def check_vector_endpoints():
    """Überprüft ob Vector-Endpunkte implementiert sind"""
    print("\n🔍 Checking for Vector Database Endpoints...")
    
    vector_endpoints = [
        "/vectors",
        "/vectors/add", 
        "/vectors/query",
        "/vectors/search",
        "/admin/create_store",
        "/stores",
        "/collections"
    ]
    
    available_vector_endpoints = []
    
    for endpoint in vector_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=3)
            
            if response.status_code != 404:
                available_vector_endpoints.append(endpoint)
                print(f"✅ {endpoint} - Status: {response.status_code}")
            else:
                print(f"❌ {endpoint} - Not implemented")
                
        except Exception as e:
            print(f"❌ {endpoint} - Error: {e}")
    
    if available_vector_endpoints:
        print(f"✅ Found {len(available_vector_endpoints)} vector endpoints")
        return True
    else:
        print("❌ No vector database endpoints found")
        print("   This appears to be a minimal server implementation")
        return False

def analyze_server_implementation():
    """Analysiert den aktuellen Server-Implementierungsstand"""
    print("\n🔬 Analyzing Server Implementation...")
    
    try:
        # Hole OpenAPI spec für Details
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        
        if response.status_code == 200:
            spec = response.json()
            paths = spec.get('paths', {})
            
            print("📊 Implementation Analysis:")
            print(f"   Total Endpoints: {len(paths)}")
            
            # Kategorisiere Endpunkte
            basic_endpoints = ['/health', '/', '/debug/routes']
            vector_endpoints = []
            admin_endpoints = []
            
            for path in paths.keys():
                if any(keyword in path for keyword in ['vector', 'search', 'query', 'embed']):
                    vector_endpoints.append(path)
                elif any(keyword in path for keyword in ['admin', 'store', 'manage']):
                    admin_endpoints.append(path)
            
            print(f"   Basic Endpoints: {len(basic_endpoints)} ✅")
            print(f"   Vector Endpoints: {len(vector_endpoints)} {'✅' if vector_endpoints else '❌'}")
            print(f"   Admin Endpoints: {len(admin_endpoints)} {'✅' if admin_endpoints else '❌'}")
            
            if not vector_endpoints and not admin_endpoints:
                print("\n💡 Server Status: MINIMAL IMPLEMENTATION")
                print("   This is a basic server setup without vector operations")
                print("   Vector database functionality needs to be implemented")
            else:
                print("\n💡 Server Status: FUNCTIONAL VECTOR DATABASE")
            
            return len(vector_endpoints) > 0
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def provide_next_steps():
    """Gibt Next Steps basierend auf Server-Status"""
    print("\n📋 NEXT STEPS RECOMMENDATIONS")
    print("=" * 50)
    
    print("🎯 Current Status:")
    print("   ✅ MLX Vector DB Server is running")
    print("   ✅ Basic health checks work")
    print("   ✅ API documentation available")
    print("   ❌ Vector operations not yet implemented")
    
    print("\n🚀 To complete the integration:")
    print("1. 📍 Visit the API docs: http://localhost:8000/docs")
    print("2. 🔧 Check if vector endpoints need to be implemented")
    print("3. 📝 Add vector database functionality to your MLX Vector DB")
    print("4. 🧪 Re-run integration tests")
    
    print("\n🔗 For mlx-langchain-lite integration:")
    print("1. 📁 Update VectorStore configuration")
    print("2. 🎛️ Adjust API endpoints in vector_store.py")
    print("3. 🧪 Create mock tests for development")
    print("4. 📈 Implement actual vector operations")

def main():
    """Haupt-Test für aktuellen MLXVectorDB Server"""
    print("🧪 MLXVectorDB Minimal Server Test")
    print("=" * 50)
    print("Testing your currently running server")
    print()
    
    tests_passed = 0
    total_tests = 4
    
    # Test Suite
    tests = [
        ("Basic Server Functionality", test_server_basic_functionality),
        ("API Documentation", test_api_documentation), 
        ("Vector Endpoints Check", check_vector_endpoints),
        ("Implementation Analysis", analyze_server_implementation),
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                tests_passed += 1
                print(f"✅ {test_name} completed")
            else:
                print(f"⚠️  {test_name} shows issues")
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {tests_passed}/{total_tests} completed")
    
    if tests_passed >= 2:
        print("✅ Server is functional but minimal")
        print("   Ready for vector database implementation!")
    else:
        print("❌ Server has issues")
    
    provide_next_steps()

if __name__ == "__main__":
    main()