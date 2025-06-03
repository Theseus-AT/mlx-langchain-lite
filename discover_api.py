#!/usr/bin/env python3
"""
API Discovery Tool für MLXVectorDB
Findet die korrekten Endpunkte deines laufenden Servers
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_common_endpoints():
    """Testet häufige API-Endpunkte"""
    print("🔍 Discovering MLXVectorDB API Endpoints...")
    
    # Liste möglicher Endpunkte
    endpoints_to_test = [
        "/",
        "/health",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/status",
        "/api/health",
        "/api/status", 
        "/vectors",
        "/admin",
        "/monitoring/health",
        "/performance/health",
        "/admin/create_store",
        "/vectors/add",
        "/vectors/query",
        "/debug/routes"
    ]
    
    working_endpoints = []
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=3)
            
            if response.status_code == 200:
                print(f"✅ {endpoint} - Status: {response.status_code}")
                working_endpoints.append(endpoint)
                
                # Zeige ersten Teil der Antwort
                try:
                    if endpoint == "/openapi.json":
                        data = response.json()
                        print(f"   📋 OpenAPI spec found - {len(data.get('paths', {}))} endpoints")
                    elif "application/json" in response.headers.get("content-type", ""):
                        data = response.json()
                        print(f"   📄 JSON response: {str(data)[:100]}...")
                    else:
                        text = response.text[:100]
                        print(f"   📄 Response: {text}...")
                except:
                    print(f"   📄 Response length: {len(response.text)} chars")
                    
            elif response.status_code == 404:
                print(f"❌ {endpoint} - Not Found")
            else:
                print(f"⚠️  {endpoint} - Status: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint} - Connection error: {e}")
    
    return working_endpoints

def get_openapi_spec():
    """Holt die OpenAPI-Spezifikation"""
    print("\n📋 Fetching OpenAPI Specification...")
    
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        
        if response.status_code == 200:
            spec = response.json()
            
            print("✅ OpenAPI Spec found!")
            print(f"   Title: {spec.get('info', {}).get('title', 'unknown')}")
            print(f"   Version: {spec.get('info', {}).get('version', 'unknown')}")
            
            # Zeige alle verfügbaren Endpunkte
            paths = spec.get('paths', {})
            print(f"\n🛣️  Available Endpoints ({len(paths)}):")
            
            for path, methods in paths.items():
                method_list = list(methods.keys())
                print(f"   {path} - {', '.join(method_list).upper()}")
            
            return spec
        else:
            print(f"❌ OpenAPI spec not available: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to get OpenAPI spec: {e}")
        return None

def test_alternative_health_endpoints():
    """Testet alternative Health-Check Endpunkte"""
    print("\n🏥 Testing Health Check Endpoints...")
    
    health_endpoints = [
        "/health",
        "/healthz", 
        "/ping",
        "/status",
        "/api/health",
        "/api/v1/health",
        "/monitoring/health",
        "/system/health"
    ]
    
    for endpoint in health_endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=3)
            
            if response.status_code == 200:
                print(f"✅ Health endpoint found: {endpoint}")
                try:
                    data = response.json()
                    print(f"   📊 Response: {data}")
                    return endpoint
                except:
                    print(f"   📄 Response: {response.text[:100]}")
                    return endpoint
                    
        except Exception as e:
            continue
    
    print("❌ No standard health endpoint found")
    return None

def test_root_endpoint():
    """Testet Root-Endpunkt für Infos"""
    print("\n🏠 Testing Root Endpoint...")
    
    try:
        response = requests.get(BASE_URL, timeout=5)
        
        print(f"Root endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            content_type = response.headers.get("content-type", "")
            
            if "application/json" in content_type:
                try:
                    data = response.json()
                    print(f"✅ JSON response: {data}")
                    return data
                except:
                    pass
            
            print(f"📄 HTML/Text response ({len(response.text)} chars)")
            print(f"   Preview: {response.text[:200]}...")
            
            # Suche nach API-Dokumentation Links
            text = response.text.lower()
            if "/docs" in text:
                print("   🔍 Found /docs reference in response")
            if "/redoc" in text:
                print("   🔍 Found /redoc reference in response")
            if "api" in text:
                print("   🔍 Found 'api' references in response")
                
        return response.text
        
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return None

def main():
    """Haupt-Discovery-Funktion"""
    print("🕵️ MLXVectorDB API Discovery Tool")
    print("=" * 50)
    print(f"Target: {BASE_URL}")
    print()
    
    # 1. Test basic connectivity
    try:
        response = requests.get(BASE_URL, timeout=5)
        print(f"✅ Server is reachable (Status: {response.status_code})")
    except:
        print(f"❌ Cannot reach server at {BASE_URL}")
        return
    
    # 2. Discover endpoints
    working_endpoints = test_common_endpoints()
    
    # 3. Get OpenAPI spec if available
    openapi_spec = get_openapi_spec()
    
    # 4. Test health endpoints
    health_endpoint = test_alternative_health_endpoints()
    
    # 5. Analyze root endpoint
    root_data = test_root_endpoint()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 DISCOVERY SUMMARY")
    print("=" * 50)
    
    if working_endpoints:
        print(f"✅ Found {len(working_endpoints)} working endpoints:")
        for endpoint in working_endpoints:
            print(f"   - {endpoint}")
    else:
        print("❌ No standard endpoints found")
    
    if health_endpoint:
        print(f"✅ Health check endpoint: {health_endpoint}")
    
    if openapi_spec:
        print("✅ OpenAPI documentation available at /docs")
    
    print(f"\n🌐 Check your API documentation at: {BASE_URL}/docs")
    print(f"🔧 Alternative docs at: {BASE_URL}/redoc")
    
    return working_endpoints, openapi_spec

if __name__ == "__main__":
    main()