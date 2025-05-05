# mlx-langchain-lite

**mlx-langchain-lite** ist ein leichtgewichtiges, vollständig lokales RAG-Modul für MLX-basierte KI-Systeme.  
Es ersetzt LangChain für spezifische Anwendungsfälle in KMU-nahen KI-Backends wie "Theseus-TeamMind".

---

## 🧭 Projektziel

Entwicklung eines modularen RAG-Systems mit folgenden Eigenschaften:

- 💻 Lokal ausführbar (kompatibel mit Apple Silicon und MLX)
- 🧠 Inferenz mit MLX-basierten Text-Encodern & LLMs
- 🧩 Unterstützung von Multi-User- und Agenten-Szenarien
- 📎 Indexierung von PDFs oder Dokument-Chunks
- 🚀 Batchfähige Anfrageverarbeitung (RAG mit mehreren Prompts)
- ❌ Kein Einsatz von LangChain oder Cloud-Diensten

---

## 🗂️ Bisher enthaltene Module

| Datei                   | Funktion                                                |
|-------------------------|---------------------------------------------------------|
| `embedding_engine.py`   | Einbettung von Texten über MLX-kompatible Encoder       |
| `vector_store.py`       | Verwaltung und Abfrage von FAISS-basierten Indizes      |
| `rag_handler.py`        | RAG-Abfrage, Prompt-Zusammenbau                         |
| `batch_dispatcher.py`   | Batchverarbeitung paralleler Anfragen                   |
| `LICENSE`               | Apache 2.0 Lizenz                                       |
| `README.md`             | Diese Datei                                             |

---

## 📌 Wiedereinstieg bei späterer Session

Wenn du in einem neuen Chat-Fenster mit ChatGPT weitermachen willst:

1. Öffne dieses Projektverzeichnis.
2. Lade alle `.py`-Dateien und diese `README.md` hoch.
3. Schreibe:  
   **"Lass uns mit `mlx-langchain-lite` weitermachen, hier sind die bisherigen Dateien und das README."**

Dann erkennt ChatGPT automatisch, worum es geht, und du kannst mit der Implementierung fortfahren.

---

## 📋 Nächste geplante Schritte

- [ ] Implementierung von `embedding_engine.py` (MLX-Encoder laden & anwenden)
- [ ] Erstellung einer Nutzer-basierten Vektorstruktur in `vector_store.py`
- [ ] Integration eines konfigurierbaren FAISS-Index pro Nutzer / Agent
- [ ] Anbindung an `Theseus-TeamMind` API (`/chat/rag`, `/data/upload/pdf`)
- [ ] Optional: PII-Filterung vorschalten (z. B. via Presidio)

---

## 🧠 Optional verwendbare MLX-Encoder

- `mlx-community/gte-small`
- `mlx-community/all-MiniLM-L6-v2`
- Lokale Konvertierung von GGUF → MLX über eigenes Tooling

---

## 🧑‍💻 Projektstatus

**Stand:** Grundstruktur und Projektziel definiert  
**Bereit für:** erste Modellanbindung und Embedding-Test

---

## Lizenz

Dieses Projekt steht unter der [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---
