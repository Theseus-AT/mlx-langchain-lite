# mlx-langchain-lite

**mlx-langchain-lite** ist ein leichtgewichtiges, vollständig lokales RAG-Modul für MLX-basierte KI-Systeme.
Es ersetzt LangChain für spezifische Anwendungsfälle in KMU-nahen KI-Backends wie "Theseus-TeamMind".

---

## 🧭 Projektziel

Entwicklung eines modularen RAG-Systems mit folgenden Eigenschaften:

- 💻 Lokal ausführbar (kompatibel mit Apple Silicon und MLX)
- 🧠 Inferenz mit MLX-basierten Text-Encodern & LLMs
- 🧩 Unterstützung von Multi-User- und Agenten-Szenarien
- 📎 Indexierung von PDFs, Dokument-Chunks und Code-Repositories
- 🚀 Batchfähige Anfrageverarbeitung (RAG mit mehreren Prompts)
- 🕸️ Integrierte Web-Recherche-Funktionen
- ❌ Kein Einsatz von LangChain oder Cloud-Diensten

---

## 🗂️ Enthaltene Module

| Datei                                  | Funktion                                                                                                |
|----------------------------------------|---------------------------------------------------------------------------------------------------------|
| `mlx_components/embedding_engine.py`   | Einbettung von Texten über MLX-kompatible Encoder, Caching, Batch-Verarbeitung.                   |
| `mlx_components/vector_store.py`       | Schnittstelle zur `mlx-vector-db` für Verwaltung und Abfrage von Vektor-Indizes, Multi-User-Support. |
| `mlx_components/llm_handler.py`        | Inferenz mit MLX-basierten LLMs (z.B. Gemma) via `mlx_parallm`, Batch-Verarbeitung, Caching, RAG-Prompting. |
| `mlx_components/rerank_engine.py`      | Neuordnung von Suchergebnissen zur Relevanzsteigerung, diverse Algorithmen.                        |
| `mlx_components/rag_orchestrator.py`   | Zentraler Koordinator der RAG-Pipeline, steuert alle Komponenten, unterstützt verschiedene RAG-Modi.  |
| `tools/document_processor.py`          | Verarbeitung diverser Dokumentformate (PDF, MD, DOCX etc.), Chunking, Metadatenextraktion.         |
| `tools/code_analyzer.py`               | Analyse von Code-Repositories (Struktur, Komplexität, Elemente), Embedding von Code-Fragmenten.   |
| `tools/research_assistant.py`          | Durchführung von Web-Recherchen, Extraktion von Webinhalten, Antwortsynthese.                   |
| `LICENSE`                              | Apache 2.0 Lizenz                                                                                       |
| `README.md`                            | Diese Datei                                                                                             |

---

## 🛠️ Setup & Installation

1.  **Python-Abhängigkeiten:** Stellen Sie sicher, dass alle erforderlichen Python-Bibliotheken installiert sind. Eine `requirements.txt` oder `pyproject.toml` (empfohlen) sollte dem Projekt beigefügt werden. Wichtige Bibliotheken sind u.a.:
    * `mlx`, `mlx-lm`, `mlx_parallm`
    * `aiohttp`, `beautifulsoup4`, `newspaper3k`, `feedparser`, `selenium` (für `research_assistant`)
    * `pymupdf` (fitz), `python-docx` (mammoth), `markdown`, `pandas` (für `document_processor`)
    * `lizard`, `tree_sitter` (für `code_analyzer`)
    * `numpy`

2.  **MLX Vector DB:** Dieses Projekt benötigt eine laufende Instanz von `mlx-vector-db`. Konfigurieren Sie die `base_url` in `VectorStoreConfig` (`mlx_components/vector_store.py`) entsprechend.

3.  **Tree-sitter Grammatiken:** Für die Code-Analyse mit `tools/code_analyzer.py` müssen die `tree-sitter`-Grammatiken für die gewünschten Sprachen kompiliert und im konfigurierten Verzeichnis (`tree_sitter_grammar_dir` in `CodeAnalysisConfig`) abgelegt werden. Kompilieren Sie z.B. `.so` (Linux) oder `.dylib` (macOS) Dateien.

4.  **Selenium WebDriver:** Falls die JavaScript-basierte Web-Extraktion im `tools/research_assistant.py` genutzt wird (`enable_javascript = True`), muss ein passender WebDriver (z.B. `chromedriver`) installiert und ggf. der Pfad in `ResearchConfig` spezifiziert werden.

---

## 📌 Wiedereinstieg bei späterer Session

Wenn du in einem neuen Chat-Fenster mit ChatGPT weitermachen willst:

1.  Öffne dieses Projektverzeichnis.
2.  Lade alle `.py`-Dateien (aus `mlx_components` und `tools`) und diese `README.md` hoch.
3.  Schreibe:
    **"Lass uns mit `mlx-langchain-lite` weitermachen, hier sind die bisherigen Dateien und das README."**

Dann erkennt ChatGPT automatisch, worum es geht, und du kannst mit der Implementierung fortfahren.

---

## 📋 Nächste geplante Schritte & Verbesserungen

-   [x] Implementierung von `embedding_engine.py` (MLX-Encoder laden & anwenden) - _Größtenteils abgeschlossen._
-   [x] Nutzer-basierte Vektorstruktur in `vector_store.py` via `mlx-vector-db`.
-   [ ] **Verfeinerung der FAISS-Integration:** Klären, ob FAISS direkt (lokal) in `vector_store.py` verwaltet wird oder ob `mlx-vector-db` dies intern übernimmt und die Konfiguration entsprechend angepasst werden muss.
-   [ ] **PII-Filterung vorschalten:** Implementierung einer PII-Filterung, z.B. in `document_processor.py` oder als globaler Schritt im `rag_orchestrator.py`.
-   [ ] **API-Anbindung an `Theseus-TeamMind`:** Spezifische Implementierung der Endpunkte (`/chat/rag`, `/data/upload/pdf`).
-   [ ] **Erweitertes Konfigurationsmanagement:** Zentralisierung gemeinsamer Konfigurationen und Nutzung von Umgebungsvariablen für sensible Daten.
-   [ ] **Robustes Fehler-Logging:** Implementierung eines projektweiten, strukturierten Loggings.
-   [ ] **Testing-Framework:** Aufbau von Unit- und Integrationstests.
-   [ ] **LLM-basiertes Re-Ranking:** Vollständige Integration und Evaluierung der `rerank_with_llm_scoring` Methode aus `rerank_engine.py`.
-   [ ] **Streaming für LLM-Antworten:** Implementierung in `llm_handler.py` zur Verbesserung der User Experience.

---

## 🧠 Optional verwendbare MLX-Encoder

Eine Auswahl an Modellen, die mit `embedding_engine.py` kompatibel sind:
- `mlx-community/gte-small`
- `mlx-community/gte-large`
- `mlx-community/all-MiniLM-L6-v2`
- `mlx-community/bge-small-en`
- `mlx-community/bge-large-en`
- `mlx-community/e5-small-v2`
- `mlx-community/e5-large-v2`
- Lokale Konvertierung von GGUF → MLX über eigenes Tooling (falls benötigt).

---

## 🧑‍💻 Projektstatus

**Stand:** Kernfunktionalität der RAG-Pipeline mit diversen Tools implementiert. Die einzelnen Module sind fortgeschritten und der `rag_orchestrator.py` dient als zentrale Steuerungseinheit.
**Bereit für:** Feinschliff, Optimierung, Implementierung der verbleibenden Features (s.o.) und gründliche Tests.

---

## Lizenz

Dieses Projekt steht unter der [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---