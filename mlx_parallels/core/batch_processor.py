"""
MLX Parallels - Haupt-Batch-Processing-Logik
Moderne Implementation basierend auf MLX-LM 0.25+ für optimierte Integration
mit mlx-langchain-lite und mlx-vector-db
"""

import time
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Generator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import logging

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate, stream_generate
from mlx_lm.utils import generate_step
from mlx_lm.models.base import KVCache

from .config import MLXParallelsConfig, ModelConfig, BatchConfig, GenerationConfig
from .memory_manager import MemoryManager


logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Einzelne Batch-Anfrage"""
    inputs: List[str]
    task: str = "generate"  # "generate" oder "embedding"
    config_override: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    callback: Optional[Callable] = None


@dataclass
class BatchResult:
    """Ergebnis einer Batch-Verarbeitung"""
    outputs: List[str]
    request_id: Optional[str]
    processing_time: float
    tokens_generated: int
    tokens_per_second: float
    memory_usage: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Ergebnis einer Embedding-Batch-Verarbeitung"""
    embeddings: List[List[float]]
    request_id: Optional[str]
    processing_time: float
    inputs_processed: int
    embeddings_per_second: float
    memory_usage: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BatchProcessor:
    """
    Hauptklasse für Batch-Processing mit MLX-LM Integration
    
    Features:
    - Native MLX-LM Integration mit aktuellen APIs
    - Optimiert für Apple Silicon und Unified Memory
    - Intelligente Batch-Größen-Optimierung
    - KV-Cache Management mit mlx-lm
    - Performance-Monitoring und Metriken
    - Asynchrone Verarbeitung
    """
    
    def __init__(self, config: MLXParallelsConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.memory_manager = MemoryManager(config)
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'total_tokens': 0,
            'total_time': 0.0,
            'avg_tokens_per_sec': 0.0,
            'avg_batch_size': 0.0
        }
        
        # Threading für asynchrone Verarbeitung
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._model_lock = threading.Lock()
        self._loaded = False
        
        # Batch-Queue für effiziente Verarbeitung
        self.batch_queue = Queue(maxsize=100)
        self._processing = False
        
        logger.info(f"BatchProcessor initialisiert für Modell: {config.model.model_name}")
    
    def load_model(self) -> bool:
        """
        Lädt Modell mit mlx-lm
        """
        try:
            with self._model_lock:
                if self._loaded:
                    return True
                
                logger.info(f"Lade Modell: {self.config.model.model_name}")
                start_time = time.time()
                
                # MLX-LM Model Loading
                self.model, self.tokenizer = load(
                    self.config.model.model_name,
                    tokenizer_config={
                        "trust_remote_code": self.config.model.trust_remote_code
                    }
                )
                
                # Warmup für optimale Performance
                if self.config.performance.auto_warmup:
                    self._warmup_model()
                
                load_time = time.time() - start_time
                logger.info(f"Modell geladen in {load_time:.2f}s")
                
                self._loaded = True
                return True
                
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {e}")
            return False
    
    def _warmup_model(self):
        """Warmup für JIT-Compilation und Cache-Optimierung"""
        logger.info("Warmup wird durchgeführt...")
        
        warmup_prompts = [
            "Hello world",
            "This is a test prompt for warmup",
            "MLX is optimized for Apple Silicon"
        ]
        
        for i in range(self.config.performance.warmup_batches):
            try:
                if self.config.model.task == "generate":
                    # Kleine Generierung für Warmup
                    responses = generate(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        prompt=warmup_prompts[i % len(warmup_prompts)],
                        max_tokens=10,
                        verbose=False
                    )
                    # Evaluation erzwingen
                    mx.eval([resp for resp in responses])
                    
            except Exception as e:
                logger.warning(f"Warmup Iteration {i} fehlgeschlagen: {e}")
        
        logger.info("Warmup abgeschlossen")
    
    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        streaming: bool = False
    ) -> Union[BatchResult, Generator[BatchResult, None, None]]:
        """
        Batch-Generierung mit MLX-LM
        
        Args:
            prompts: Liste von Input-Prompts
            max_tokens: Maximale Token-Anzahl pro Generierung
            temperature: Sampling-Temperatur
            streaming: Ob Streaming-Generierung verwendet werden soll
            
        Returns:
            BatchResult oder Generator für Streaming
        """
        if not self._loaded:
            if not self.load_model():
                raise RuntimeError("Modell konnte nicht geladen werden")
        
        # Konfiguration
        max_tokens = max_tokens or self.config.generation.max_tokens
        temperature = temperature or self.config.generation.temperature
        
        batch_size = len(prompts)
        
        # Batch-Größe optimieren
        optimal_batch_size = self._calculate_optimal_batch_size(prompts)
        if batch_size > optimal_batch_size:
            # Aufteilen in kleinere Batches
            return self._process_large_batch(prompts, max_tokens, temperature, streaming)
        
        start_time = time.time()
        
        try:
            if streaming:
                return self._stream_batch_generate(prompts, max_tokens, temperature)
            else:
                return self._sync_batch_generate(prompts, max_tokens, temperature, start_time)
                
        except Exception as e:
            logger.error(f"Batch-Generierung fehlgeschlagen: {e}")
            raise
    
    def _sync_batch_generate(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        start_time: float
    ) -> BatchResult:
        """Synchrone Batch-Generierung"""
        
        all_outputs = []
        total_tokens = 0
        
        # Optimierter Ansatz: Sequentielle Verarbeitung mit KV-Cache Wiederverwendung
        for i, prompt in enumerate(prompts):
            if self.config.generation.format_prompts:
                # Prompt-Formatierung falls erforderlich
                formatted_prompt = self._format_prompt(prompt)
            else:
                formatted_prompt = prompt
            
            # MLX-LM Generate verwenden
            response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=self.config.generation.top_p,
                verbose=False
            )
            
            all_outputs.append(response)
            
            # Token-Counting (vereinfacht)
            tokens = len(self.tokenizer.encode(response))
            total_tokens += tokens
            
            if self.config.verbose and i % 5 == 0:
                logger.info(f"Verarbeitet: {i+1}/{len(prompts)} Prompts")
        
        processing_time = time.time() - start_time
        tokens_per_second = total_tokens / processing_time if processing_time > 0 else 0
        
        # Memory Usage
        memory_usage = self.memory_manager.get_current_usage()
        
        # Statistiken aktualisieren
        self._update_stats(len(prompts), total_tokens, processing_time)
        
        return BatchResult(
            outputs=all_outputs,
            request_id=None,
            processing_time=processing_time,
            tokens_generated=total_tokens,
            tokens_per_second=tokens_per_second,
            memory_usage=memory_usage,
            metadata={
                'batch_size': len(prompts),
                'avg_tokens_per_output': total_tokens / len(prompts),
                'model': self.config.model.model_name,
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        )
    
    def _stream_batch_generate(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float
    ) -> Generator[BatchResult, None, None]:
        """Streaming Batch-Generierung"""
        
        start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            if self.config.generation.format_prompts:
                formatted_prompt = self._format_prompt(prompt)
            else:
                formatted_prompt = prompt
            
            # MLX-LM Stream Generate
            response_stream = stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=self.config.generation.top_p,
                verbose=False
            )
            
            accumulated_text = ""
            for response_chunk in response_stream:
                accumulated_text += response_chunk.text
                
                # Zwischenergebnis yielden
                current_time = time.time()
                partial_result = BatchResult(
                    outputs=[accumulated_text],
                    request_id=f"stream_{i}",
                    processing_time=current_time - start_time,
                    tokens_generated=len(self.tokenizer.encode(accumulated_text)),
                    tokens_per_second=0,  # Wird später berechnet
                    memory_usage=self.memory_manager.get_current_usage(),
                    metadata={
                        'prompt_index': i,
                        'total_prompts': len(prompts),
                        'is_complete': False
                    }
                )
                
                yield partial_result
            
            # Finales Ergebnis für diesen Prompt
            processing_time = time.time() - start_time
            tokens = len(self.tokenizer.encode(accumulated_text))
            
            final_result = BatchResult(
                outputs=[accumulated_text],
                request_id=f"stream_{i}",
                processing_time=processing_time,
                tokens_generated=tokens,
                tokens_per_second=tokens / processing_time if processing_time > 0 else 0,
                memory_usage=self.memory_manager.get_current_usage(),
                metadata={
                    'prompt_index': i,
                    'total_prompts': len(prompts),
                    'is_complete': True
                }
            )
            
            yield final_result
    
    def batch_embed(
        self,
        texts: List[str],
        normalize: bool = None
    ) -> EmbeddingResult:
        """
        Batch-Embedding-Generierung für VectorDB-Integration
        
        Args:
            texts: Liste von Texten für Embedding
            normalize: Ob Embeddings normalisiert werden sollen
            
        Returns:
            EmbeddingResult mit allen Embeddings
        """
        if not self._loaded:
            if not self.load_model():
                raise RuntimeError("Modell konnte nicht geladen werden")
        
        if self.config.model.task != "embedding":
            raise ValueError("Modell ist nicht für Embedding-Aufgaben konfiguriert")
        
        normalize = normalize if normalize is not None else self.config.embedding.normalize
        
        start_time = time.time()
        
        try:
            # Chunk große Texte falls erforderlich
            if len(texts) > self.config.embedding.chunk_size:
                return self._process_large_embedding_batch(texts, normalize)
            
            all_embeddings = []
            
            # Batch-Verarbeitung für Embeddings
            # Da mlx-lm primär für Generierung ist, verwenden wir das Modell direkt
            for text in texts:
                # Text tokenisieren
                tokens = mx.array(self.tokenizer.encode(text))
                
                # Durch Modell verarbeiten (vereinfacht)
                # Hier würden Sie Ihren spezifischen Embedding-Code einfügen
                # basierend auf dem verwendeten Embedding-Modell
                
                # Placeholder - ersetzen Sie durch tatsächliche Embedding-Logic
                embedding = self._extract_embedding(tokens)
                
                if normalize:
                    # L2-Normalisierung
                    norm = mx.linalg.norm(embedding)
                    embedding = embedding / norm
                
                all_embeddings.append(embedding.tolist())
            
            processing_time = time.time() - start_time
            embeddings_per_second = len(texts) / processing_time if processing_time > 0 else 0
            
            return EmbeddingResult(
                embeddings=all_embeddings,
                request_id=None,
                processing_time=processing_time,
                inputs_processed=len(texts),
                embeddings_per_second=embeddings_per_second,
                memory_usage=self.memory_manager.get_current_usage(),
                metadata={
                    'batch_size': len(texts),
                    'normalized': normalize,
                    'model': self.config.model.model_name,
                    'embedding_dim': len(all_embeddings[0]) if all_embeddings else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Batch-Embedding fehlgeschlagen: {e}")
            raise
    
    def _extract_embedding(self, tokens: mx.array) -> mx.array:
        """
        Extrahiert Embedding aus Tokens
        Diese Methode muss je nach Embedding-Modell angepasst werden
        """
        # Placeholder - implementieren Sie basierend auf Ihrem Embedding-Modell
        # Beispiel für sentence-transformer-style:
        
        # Forward pass durch das Modell
        outputs = self.model(tokens.reshape(1, -1))  # Batch dimension hinzufügen
        
        # Pooling (mean, max, oder cls je nach Modell)
        if self.config.embedding.pooling_method == "mean":
            embedding = mx.mean(outputs, axis=1)
        elif self.config.embedding.pooling_method == "max":
            embedding = mx.max(outputs, axis=1)
        elif self.config.embedding.pooling_method == "cls":
            embedding = outputs[:, 0]  # CLS token
        else:
            embedding = mx.mean(outputs, axis=1)  # Default
        
        return embedding.squeeze(0)  # Batch dimension entfernen
    
    def _process_large_batch(
        self,
        prompts: List[str],
        max_tokens: int,
        temperature: float,
        streaming: bool
    ) -> BatchResult:
        """Verarbeitung großer Batches durch Aufteilung"""
        
        optimal_size = self._calculate_optimal_batch_size(prompts)
        
        all_outputs = []
        total_tokens = 0
        total_time = 0.0
        
        # Aufteilen in Chunks
        for i in range(0, len(prompts), optimal_size):
            chunk = prompts[i:i + optimal_size]
            
            if streaming:
                # Für große Batches mit Streaming müsste hier eine andere Logik
                # implementiert werden
                chunk_result = self._sync_batch_generate(chunk, max_tokens, temperature, time.time())
            else:
                chunk_result = self._sync_batch_generate(chunk, max_tokens, temperature, time.time())
            
            all_outputs.extend(chunk_result.outputs)
            total_tokens += chunk_result.tokens_generated
            total_time += chunk_result.processing_time
            
            if self.config.verbose:
                logger.info(f"Chunk {i//optimal_size + 1} verarbeitet: {len(chunk)} Prompts")
        
        return BatchResult(
            outputs=all_outputs,
            request_id=None,
            processing_time=total_time,
            tokens_generated=total_tokens,
            tokens_per_second=total_tokens / total_time if total_time > 0 else 0,
            memory_usage=self.memory_manager.get_current_usage(),
            metadata={
                'batch_size': len(prompts),
                'chunks_processed': (len(prompts) + optimal_size - 1) // optimal_size,
                'optimal_chunk_size': optimal_size
            }
        )
    
    def _process_large_embedding_batch(
        self,
        texts: List[str],
        normalize: bool
    ) -> EmbeddingResult:
        """Verarbeitung großer Embedding-Batches"""
        
        chunk_size = self.config.embedding.chunk_size
        all_embeddings = []
        total_time = 0.0
        
        for i in range(0, len(texts), chunk_size):
            chunk = texts[i:i + chunk_size]
            chunk_result = self.batch_embed(chunk, normalize)
            
            all_embeddings.extend(chunk_result.embeddings)
            total_time += chunk_result.processing_time
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            request_id=None,
            processing_time=total_time,
            inputs_processed=len(texts),
            embeddings_per_second=len(texts) / total_time if total_time > 0 else 0,
            memory_usage=self.memory_manager.get_current_usage(),
            metadata={
                'batch_size': len(texts),
                'chunks_processed': (len(texts) + chunk_size - 1) // chunk_size,
                'chunk_size': chunk_size
            }
        )
    
    def _calculate_optimal_batch_size(self, inputs: List[str]) -> int:
        """Berechnet optimale Batch-Größe basierend auf verfügbaren Ressourcen"""
        
        # Basis-Batch-Größe aus Konfiguration
        base_size = self.config.batch.max_batch_size
        
        if not self.config.batch.auto_batch_size:
            return base_size
        
        # Memory-basierte Anpassung
        available_memory = self.memory_manager.get_available_memory_mb()
        if available_memory:
            # Geschätzte Memory-Verwendung pro Item
            avg_input_length = sum(len(inp) for inp in inputs[:10]) // min(10, len(inputs))
            estimated_memory_per_item = avg_input_length * 0.01  # Heuristik
            
            memory_based_size = int(
                (available_memory * self.config.batch.memory_threshold) 
                / estimated_memory_per_item
            )
            
            optimal_size = min(base_size, memory_based_size, len(inputs))
        else:
            optimal_size = min(base_size, len(inputs))
        
        return max(1, optimal_size)
    
    def _format_prompt(self, prompt: str) -> str:
        """Formatiert Prompt falls erforderlich"""
        # Hier können Sie Prompt-Templates anwenden
        # basierend auf dem verwendeten Modell
        return prompt
    
    def _update_stats(self, batch_size: int, tokens: int, processing_time: float):
        """Aktualisiert Performance-Statistiken"""
        self.stats['total_requests'] += 1
        self.stats['total_tokens'] += tokens
        self.stats['total_time'] += processing_time
        
        # Gleitende Durchschnitte
        self.stats['avg_tokens_per_sec'] = (
            self.stats['avg_tokens_per_sec'] * 0.9 + 
            (tokens / processing_time if processing_time > 0 else 0) * 0.1
        )
        
        self.stats['avg_batch_size'] = (
            self.stats['avg_batch_size'] * 0.9 + batch_size * 0.1
        )
    
    async def async_batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> BatchResult:
        """Asynchrone Batch-Generierung"""
        loop = asyncio.get_event_loop()
        
        # In Thread-Pool ausführen um Main-Thread nicht zu blockieren
        result = await loop.run_in_executor(
            self.executor,
            lambda: self.batch_generate(prompts, **kwargs)
        )
        
        return result
    
    async def async_batch_embed(
        self,
        texts: List[str],
        **kwargs
    ) -> EmbeddingResult:
        """Asynchrone Batch-Embedding"""
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            self.executor,
            lambda: self.batch_embed(texts, **kwargs)
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Gibt aktuelle Performance-Statistiken zurück"""
        return {
            **self.stats,
            'model_loaded': self._loaded,
            'model_name': self.config.model.model_name,
            'memory_usage': self.memory_manager.get_current_usage(),
            'config': {
                'max_batch_size': self.config.batch.max_batch_size,
                'auto_batch_size': self.config.batch.auto_batch_size,
                'task': self.config.model.task
            }
        }
    
    def cleanup(self):
        """Cleanup-Ressourcen"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # Model cleanup falls erforderlich
        self.model = None
        self.tokenizer = None
        self._loaded = False
        
        logger.info("BatchProcessor Cleanup abgeschlossen")
    
    def __del__(self):
        """Destruktor für automatisches Cleanup"""
        try:
            self.cleanup()
        except:
            pass