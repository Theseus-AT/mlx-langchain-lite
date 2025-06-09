#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

"""
Quick Test Script für mlx-langchain-lite
Testet die Grundfunktionalität ohne komplexe Dependencies
"""

import os
import sys
import time
from pathlib import Path

def check_system_requirements():
    """Prüft Hardware und System-Voraussetzungen"""
    print("🔍 System Requirements Check...")
    
    # Python Version
    python_version = sys.version_info
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 9):
        print("❌ Python 3.9+ erforderlich!")
        return False
    
    # Platform Check
    import platform
    if platform.system() != "Darwin":
        print("⚠️  MLX funktioniert nur auf macOS!")
        return False
    
    # Apple Silicon Check
    try:
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                              capture_output=True, text=True)
        cpu_info = result.stdout.strip()
        if 'Apple' in cpu_info:
            print(f"✅ Apple Silicon detected: {cpu_info}")
        else:
            print(f"⚠️  Intel Mac detected: {cpu_info}")
            print("   MLX funktioniert am besten auf Apple Silicon!")
    except:
        print("⚠️  Couldn't detect CPU type")
    
    return True

def test_mlx_installation():
    """Testet MLX Installation"""
    print("\n🧪 Testing MLX Installation...")
    
    try:
        import mlx.core as mx
        print("✅ mlx.core imported successfully")
        
        # Simple MLX test
        a = mx.array([1, 2, 3])
        b = mx.array([4, 5, 6])
        c = a + b
        print(f"✅ MLX computation test: {a} + {b} = {c}")
        
    except ImportError as e:
        print(f"❌ MLX not installed: {e}")
        print("   Install with: pip install mlx")
        return False
    except Exception as e:
        print(f"❌ MLX error: {e}")
        return False
    
    try:
        import mlx_lm
        print("✅ mlx-lm imported successfully")
    except ImportError:
        print("❌ mlx-lm not installed")
        print("   Install with: pip install mlx-lm")
        return False
    
    return True

def test_embedding_model():
    """Testet ein einfaches Embedding Model"""
    print("\n🎯 Testing Embedding Model...")
    
    try:
        from mlx_lm import load
        
        # Kleines, schnelles Test-Model
        print("📥 Loading test embedding model...")
        
        # Verwende ein sehr kleines Model für Test
        model_name = "mlx-community/all-MiniLM-L6-v2"
        
        print(f"   Trying to load: {model_name}")
        start_time = time.time()
        
        # Load model (downloads if needed)
        model, tokenizer = load(model_name)
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Simple test
        test_text = "This is a test sentence."
        print(f"🧪 Testing with: '{test_text}'")
        
        # Tokenize
        tokens = tokenizer.encode(test_text)
        print(f"✅ Tokenized: {len(tokens)} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
        print("   This might be normal if models aren't downloaded yet")
        return False

def test_mlx_vector_db():
    """Testet deine eigene mlx_vector_db"""
    print("\n🗃️ Testing MLX Vector Database...")
    
    try:
        # Teste ob mlx_vector_db Module existieren
        import sys
        from pathlib import Path
        
        # Schaue nach deiner mlx_vector_db Implementation
        vector_db_paths = [
            "mlx_components/vector_store.py",
            "mlx_vector_db/",
            "vector_db/",
        ]
        
        found_vector_db = False
        for path in vector_db_paths:
            if Path(path).exists():
                print(f"✅ Found vector database module: {path}")
                found_vector_db = True
                break
        
        if not found_vector_db:
            print("❌ MLX Vector DB module not found")
            print("   Expected paths:", vector_db_paths)
            return False
        
        # Versuche VectorStore zu importieren
        try:
            sys.path.insert(0, str(Path.cwd()))
            from mlx_components.vector_store import MLXVectorStore
            print("✅ MLXVectorStore imported successfully")
            
            # Test basic initialization
            # (ohne tatsächliche Verbindung, da Server evtl. nicht läuft)
            print("✅ MLXVectorStore class available for testing")
            
        except ImportError as e:
            print(f"⚠️  MLXVectorStore import failed: {e}")
            print("   This is normal if mlx_vector_db server is not running")
            
        # Teste ob Vector DB Server läuft
        try:
            import requests
            
            # Standard-URL aus der Config (falls verfügbar)
            test_urls = [
                "http://localhost:8000",
                "http://localhost:8080", 
                "http://127.0.0.1:8000"
            ]
            
            for url in test_urls:
                try:
                    response = requests.get(f"{url}/health", timeout=2)
                    if response.status_code == 200:
                        print(f"✅ MLX Vector DB server running at: {url}")
                        return True
                except:
                    continue
            
            print("⚠️  MLX Vector DB server not running")
            print("   Start your mlx_vector_db server first")
            print("   Testing will continue without server connection")
            
        except ImportError:
            print("⚠️  requests not installed for server testing")
            
        return True
        
    except Exception as e:
        print(f"❌ MLX Vector DB test failed: {e}")
        return False

def test_document_processing():
    """Testet Document Processing Dependencies"""
    print("\n📄 Testing Document Processing...")
    
    # Test PyMuPDF (für PDFs)
    try:
        import fitz  # PyMuPDF
        print("✅ PyMuPDF (PDF processing) available")
    except ImportError:
        print("❌ PyMuPDF not installed: pip install PyMuPDF")
    
    # Test python-docx
    try:
        import docx
        print("✅ python-docx (Word processing) available")
    except ImportError:
        print("❌ python-docx not installed: pip install python-docx")
    
    # Test markdown
    try:
        import markdown
        print("✅ markdown processing available")
    except ImportError:
        print("❌ markdown not installed: pip install markdown")

def test_configuration_system():
    """Testet das Configuration System"""
    print("\n⚙️ Testing Configuration System...")
    
    # Check if config module exists
    config_path = Path("config/system_config.py")
    if config_path.exists():
        print(f"✅ Configuration module found: {config_path}")
        
        try:
            # Import test
            sys.path.insert(0, str(Path.cwd()))
            from config.system_config import SystemConfig
            print("✅ SystemConfig class imported successfully")
            
            # Create test config
            config = SystemConfig()
            print(f"✅ Default config created")
            
            return True
            
        except ImportError as e:
            print(f"❌ Config import failed: {e}")
            return False
    else:
        print(f"❌ Configuration module not found: {config_path}")
        print("   Make sure you're in the mlx-langchain-lite directory")
        return False

def main():
    """Haupt-Test-Funktion"""
    print("🚀 MLX-LangChain-Lite Quick Test")
    print("=" * 50)
    
    all_tests_passed = True
    
    # 1. System Requirements
    if not check_system_requirements():
        all_tests_passed = False
    
    # 2. MLX Installation
    if not test_mlx_installation():
        all_tests_passed = False
    
    # 3. Configuration System
    if not test_configuration_system():
        all_tests_passed = False
    
    # 4. MLX Vector DB (deine eigene)
    if not test_mlx_vector_db():
        all_tests_passed = False
    
    # 5. Document Processing
    test_document_processing()  # Warning only
    
    # 6. Embedding Model (optional, might take time)
    print("\n❓ Test embedding model? (downloads ~100MB, takes time)")
    response = input("y/N: ").lower().strip()
    if response == 'y':
        test_embedding_model()
    
    print("\n" + "=" * 50)
    if all_tests_passed:
        print("🎉 Grundlegende Tests bestanden!")
        print("   Du kannst mit der Entwicklung beginnen!")
    else:
        print("⚠️  Einige Tests fehlgeschlagen")
        print("   Installiere fehlende Dependencies")
    
    print("\n📚 Next Steps:")
    print("1. pip install -r requirements.txt")
    print("2. Erstelle test_documents/ Ordner mit PDFs")
    print("3. Starte mit: python quick_test.py")

if __name__ == "__main__":
    main()