#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

"""
MLX Parallels Quick Setup Script
Automatische Installation und Integration für mlx-langchain-lite
"""

import os
import sys
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLXParallelsSetup:
    """Setup-Klasse für MLX Parallels Integration"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.mlx_parallels_dir = self.project_root / "mlx_parallels"
        
    def check_requirements(self) -> Dict[str, bool]:
        """Überprüft System-Anforderungen"""
        requirements = {
            'python_version': sys.version_info >= (3, 9),
            'apple_silicon': False,
            'mlx_available': False,
            'mlx_lm_available': False,
            'project_structure': False
        }
        
        # Apple Silicon Check
        try:
            import platform
            processor = platform.processor().lower()
            machine = platform.machine().lower()
            requirements['apple_silicon'] = (
                'arm' in processor or 'apple' in processor or 
                'aarch64' in machine or 'arm64' in machine
            )
        except:
            pass
        
        # MLX Check
        try:
            import mlx.core as mx
            requirements['mlx_available'] = True
            
            # Test functionality
            test_array = mx.random.normal((10, 10))
            mx.eval(test_array)
            
        except ImportError:
            requirements['mlx_available'] = False
        except Exception as e:
            logger.warning(f"MLX funktional test fehlgeschlagen: {e}")
        
        # MLX-LM Check
        try:
            import mlx_lm
            requirements['mlx_lm_available'] = True
        except ImportError:
            requirements['mlx_lm_available'] = False
        
        # Project Structure Check
        mlx_components = self.project_root / "mlx_components"
        requirements['project_structure'] = (
            mlx_components.exists() and
            (mlx_components / "llm_handler.py").exists() and
            (mlx_components / "rag_orchestrator.py").exists()
        )
        
        return requirements
    
    def install_dependencies(self) -> bool:
        """Installiert erforderliche Dependencies"""
        logger.info("🔧 Installiere Dependencies...")
        
        dependencies = [
            "mlx>=0.25.2",
            "mlx-lm>=0.20.0", 
            "numpy>=1.24.0",
            "psutil>=5.9.0",
            "aiohttp>=3.8.0",
            "structlog>=23.1.0"
        ]
        
        try:
            for dep in dependencies:
                logger.info(f"   Installiere {dep}...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", dep],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    logger.error(f"❌ Fehler bei Installation von {dep}: {result.stderr}")
                    return False
            
            logger.info("✅ Alle Dependencies installiert")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency-Installation fehlgeschlagen: {e}")
            return False
    
    def create_mlx_parallels_structure(self) -> bool:
        """Erstellt MLX Parallels Verzeichnisstruktur"""
        logger.info("📁 Erstelle MLX Parallels Struktur...")
        
        directories = [
            self.mlx_parallels_dir,
            self.mlx_parallels_dir / "core",
            self.mlx_parallels_dir / "integration",
            self.mlx_parallels_dir / "examples",
            self.mlx_parallels_dir / "tests"
        ]
        
        try:
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"   ✅ {directory}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Struktur-Erstellung fehlgeschlagen: {e}")
            return False
    
    def create_init_files(self) -> bool:
        """Erstellt __init__.py Dateien"""
        logger.info("📝 Erstelle __init__.py Dateien...")
        
        init_files = {
            self.mlx_parallels_dir / "__init__.py": '''"""
MLX Parallels - Batch-Processing für Apple Silicon
"""
__version__ = "0.1.0"

from .core.config import MLXParallelsConfig, get_fast_inference_config
from .core.batch_processor import BatchProcessor, BatchResult

__all__ = ['MLXParallelsConfig', 'BatchProcessor', 'BatchResult', 'get_fast_inference_config']
''',
            self.mlx_parallels_dir / "core" / "__init__.py": '''"""
MLX Parallels Core Module
"""
from .config import *
from .batch_processor import *
from .memory_manager import *

__all__ = ['MLXParallelsConfig', 'BatchProcessor', 'BatchResult', 'MemoryManager']
''',
            self.mlx_parallels_dir / "integration" / "__init__.py": '''"""
MLX Parallels Integration Module
"""
''',
            self.mlx_parallels_dir / "examples" / "__init__.py": '''"""
MLX Parallels Examples
"""
''',
        }
        
        try:
            for file_path, content in init_files.items():
                with open(file_path, 'w') as f:
                    f.write(content)
                logger.info(f"   ✅ {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ __init__.py Erstellung fehlgeschlagen: {e}")
            return False
    
    def backup_existing_files(self) -> bool:
        """Erstellt Backup von bestehenden Dateien"""
        logger.info("💾 Erstelle Backup der bestehenden Dateien...")
        
        files_to_backup = [
            self.project_root / "mlx_components" / "llm_handler.py",
            self.project_root / "mlx_components" / "rag_orchestrator.py"
        ]
        
        try:
            backup_dir = self.project_root / "backup_before_mlx_parallels"
            backup_dir.mkdir(exist_ok=True)
            
            for file_path in files_to_backup:
                if file_path.exists():
                    backup_path = backup_dir / file_path.name
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"   ✅ Backup: {file_path.name} -> {backup_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup fehlgeschlagen: {e}")
            return False
    
    def download_model_for_testing(self) -> bool:
        """Lädt Test-Modell herunter"""
        logger.info("📥 Lade Test-Modell herunter (Llama 3.2 3B)...")
        
        try:
            # Test ob mlx-lm verfügbar ist
            from mlx_lm import load
            
            model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"
            logger.info(f"   Lade {model_name}...")
            
            # Model laden (wird automatisch heruntergeladen falls nicht vorhanden)
            model, tokenizer = load(model_name)
            
            # Schneller Test
            logger.info("   Teste Modell...")
            from mlx_lm import generate
            test_response = generate(
                model=model,
                tokenizer=tokenizer,
                prompt="Hello",
                max_tokens=5,
                temp=0.1,
                verbose=False
            )
            
            logger.info(f"   ✅ Modell funktioniert: '{test_response.strip()}'")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Modell-Download/Test fehlgeschlagen: {e}")
            logger.info("   Das ist nicht kritisch - Sie können später ein Modell laden")
            return True  # Nicht kritisch für Setup
    
    async def test_integration(self) -> bool:
        """Testet die Integration"""
        logger.info("🧪 Teste MLX Parallels Integration...")
        
        try:
            # Import Test
            sys.path.insert(0, str(self.project_root))
            
            from mlx_parallels.core.config import get_fast_inference_config
            from mlx_parallels.core.batch_processor import BatchProcessor
            
            # Config Test
            config = get_fast_inference_config("mlx-community/Llama-3.2-3B-Instruct-4bit")
            logger.info("   ✅ Config erstellt")
            
            # Processor Test
            processor = BatchProcessor(config)
            logger.info("   ✅ BatchProcessor erstellt")
            
            # Model Loading Test (optional)
            try:
                success = processor.load_model()
                if success:
                    logger.info("   ✅ Modell geladen")
                    
                    # Quick Generation Test
                    result = processor.batch_generate(["Test"], max_tokens=5)
                    if result.outputs and result.outputs[0].strip():
                        logger.info(f"   ✅ Generierung funktioniert: '{result.outputs[0].strip()}'")
                    else:
                        logger.warning("   ⚠️  Generierung lieferte leere Antwort")
                else:
                    logger.warning("   ⚠️  Modell-Loading fehlgeschlagen")
            except Exception as model_error:
                logger.warning(f"   ⚠️  Modell-Test fehlgeschlagen: {model_error}")
            finally:
                processor.cleanup()
            
            logger.info("✅ Integration-Test abgeschlossen")
            return True
            
        except Exception as e:
            logger.error(f"❌ Integration-Test fehlgeschlagen: {e}")
            return False
    
    def create_example_usage(self) -> bool:
        """Erstellt Beispiel-Nutzung"""
        logger.info("📄 Erstelle Beispiel-Dateien...")
        
        example_code = '''#!/usr/bin/env python3
"""
MLX Parallels - Schnellstart Beispiel
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mlx_parallels.core.config import get_fast_inference_config
from mlx_parallels.core.batch_processor import BatchProcessor


async def main():
    print("🚀 MLX Parallels Schnellstart")
    
    # 1. Konfiguration erstellen
    config = get_fast_inference_config("mlx-community/Llama-3.2-3B-Instruct-4bit")
    print("✅ Konfiguration erstellt")
    
    # 2. Batch Processor erstellen
    processor = BatchProcessor(config)
    print("✅ Batch Processor erstellt")
    
    # 3. Modell laden
    print("📥 Lade Modell... (kann beim ersten Mal dauern)")
    success = processor.load_model()
    
    if not success:
        print("❌ Modell konnte nicht geladen werden")
        return
    
    print("✅ Modell geladen")
    
    # 4. Test-Prompts
    test_prompts = [
        "Was ist MLX?",
        "Erkläre Apple Silicon",
        "Vorteile von Unified Memory"
    ]
    
    # 5. Batch-Generierung
    print(f"🔄 Generiere Antworten für {len(test_prompts)} Prompts...")
    
    result = processor.batch_generate(
        prompts=test_prompts,
        max_tokens=50,
        temperature=0.7
    )
    
    # 6. Ergebnisse anzeigen
    print(f"\\n✅ Fertig! {result.tokens_per_second:.1f} tokens/sec")
    print("-" * 60)
    
    for i, (prompt, response) in enumerate(zip(test_prompts, result.outputs)):
        print(f"\\n{i+1}. {prompt}")
        print(f"   → {response.strip()}")
    
    # 7. Cleanup
    processor.cleanup()
    print("\\n🎉 Beispiel abgeschlossen!")


if __name__ == "__main__":
    asyncio.run(main())
'''
        
        try:
            example_file = self.mlx_parallels_dir / "examples" / "quickstart.py"
            with open(example_file, 'w') as f:
                f.write(example_code)
            
            # Make executable
            os.chmod(example_file, 0o755)
            
            logger.info(f"   ✅ {example_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Beispiel-Erstellung fehlgeschlagen: {e}")
            return False
    
    def print_next_steps(self):
        """Zeigt nächste Schritte an"""
        print("\n" + "="*60)
        print("🎉 MLX Parallels Setup abgeschlossen!")
        print("="*60)
        
        print("\n📋 Nächste Schritte:")
        print("\n1. 📄 INTEGRATION STARTEN:")
        print("   Kopieren Sie die korrigierten Code-Snippets in Ihre Dateien:")
        print("   • mlx_components/llm_handler.py erweitern")
        print("   • mlx_components/rag_orchestrator.py erweitern")
        
        print("\n2. 🧪 TESTEN:")
        print(f"   python {self.mlx_parallels_dir}/examples/quickstart.py")
        
        print("\n3. 📚 VERWENDUNG in Ihrem Code:")
        print("""
   # Ihr bestehender Code funktioniert weiter:
   response = await llm_handler.generate_response("Hello World")
   
   # Neue Batch-API für bessere Performance:
   responses = await llm_handler.batch_inference([
       "Prompt 1", "Prompt 2", "Prompt 3"
   ])
   
   # RAG-Batch für Ihren Orchestrator:
   rag_results = await rag_orchestrator.batch_rag_query([
       "Query 1", "Query 2", "Query 3"
   ])""")
        
        print("\n4. 📁 DATEIEN:")
        print(f"   • Backup Ihrer Originaldateien: {self.project_root}/backup_before_mlx_parallels/")
        print(f"   • MLX Parallels Code: {self.mlx_parallels_dir}/")
        print(f"   • Beispiele: {self.mlx_parallels_dir}/examples/")
        
        print("\n5. 🔧 BEI PROBLEMEN:")
        print("   • Überprüfen Sie die Requirements mit check_requirements()")
        print("   • Schauen Sie in die Backup-Dateien falls etwas schief geht")
        print("   • Testen Sie einzelne Komponenten isoliert")
        
        print("\n✨ Happy Coding mit MLX Parallels!")


async def main():
    """Hauptfunktion für Setup"""
    print("🚀 MLX Parallels Setup gestartet")
    print("="*50)
    
    # Projekt-Root ermitteln
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = os.getcwd()
    
    print(f"📁 Projekt-Verzeichnis: {project_root}")
    
    # Setup-Instanz erstellen
    setup = MLXParallelsSetup(project_root)
    
    # 1. Requirements checken
    print("\n1️⃣ Überprüfe System-Anforderungen...")
    requirements = setup.check_requirements()
    
    for req, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {req}: {status}")
    
    # Kritische Requirements prüfen
    if not requirements['python_version']:
        print("❌ Python 3.9+ erforderlich")
        return False
    
    if not requirements['project_structure']:
        print("❌ mlx-langchain-lite Struktur nicht gefunden")
        print("   Stellen Sie sicher, dass Sie im richtigen Verzeichnis sind")
        return False
    
    if not requirements['apple_silicon']:
        print("⚠️  Kein Apple Silicon erkannt - Performance könnte reduziert sein")
    
    # 2. Dependencies installieren
    if not requirements['mlx_available'] or not requirements['mlx_lm_available']:
        print("\n2️⃣ Installiere fehlende Dependencies...")
        if not setup.install_dependencies():
            return False
    else:
        print("\n2️⃣ ✅ Dependencies bereits installiert")
    
    # 3. Backup erstellen
    print("\n3️⃣ Erstelle Backup der bestehenden Dateien...")
    if not setup.backup_existing_files():
        return False
    
    # 4. MLX Parallels Struktur erstellen
    print("\n4️⃣ Erstelle MLX Parallels Struktur...")
    if not setup.create_mlx_parallels_structure():
        return False
    
    # 5. __init__.py Dateien erstellen
    print("\n5️⃣ Erstelle Modul-Dateien...")
    if not setup.create_init_files():
        return False
    
    # 6. Test-Modell herunterladen
    print("\n6️⃣ Lade Test-Modell...")
    setup.download_model_for_testing()
    
    # 7. Beispiele erstellen
    print("\n7️⃣ Erstelle Beispiel-Dateien...")
    if not setup.create_example_usage():
        return False
    
    # 8. Integration testen
    print("\n8️⃣ Teste Integration...")
    await setup.test_integration()
    
    # 9. Nächste Schritte anzeigen
    setup.print_next_steps()
    
    return True


def check_requirements_only():
    """Nur Requirements checken ohne Installation"""
    setup = MLXParallelsSetup(os.getcwd())
    requirements = setup.check_requirements()
    
    print("🔍 MLX Parallels Requirements Check")
    print("="*40)
    
    for req, status in requirements.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {req}: {status}")
    
    if all(requirements.values()):
        print("\n🎉 Alle Requirements erfüllt!")
        return True
    else:
        print("\n⚠️  Einige Requirements nicht erfüllt")
        
        if not requirements['python_version']:
            print("   → Aktualisieren Sie auf Python 3.9+")
        if not requirements['apple_silicon']:
            print("   → MLX funktioniert am besten auf Apple Silicon")
        if not requirements['mlx_available']:
            print("   → Installieren Sie MLX: pip install mlx>=0.25.2")
        if not requirements['mlx_lm_available']:
            print("   → Installieren Sie MLX-LM: pip install mlx-lm>=0.20.0")
        if not requirements['project_structure']:
            print("   → Stellen Sie sicher, dass Sie im mlx-langchain-lite Verzeichnis sind")
        
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        # Nur Requirements checken
        check_requirements_only()
    else:
        # Vollständiges Setup
        try:
            success = asyncio.run(main())
            if success:
                print("\n🎉 Setup erfolgreich abgeschlossen!")
                sys.exit(0)
            else:
                print("\n❌ Setup fehlgeschlagen")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\n⚠️  Setup abgebrochen")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unerwarteter Fehler: {e}")
            sys.exit(1)