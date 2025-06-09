#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

"""
MLX Embedding Engine - Enhanced mit MLX Parallels Integration
Batch-Embedding-Generierung für optimale Performance mit mlx-vector-db
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import hashlib
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

# MLX Parallels Integration
try:
    from mlx_parallels.core.batch_processor import BatchProcessor, EmbeddingResult
    from mlx_parallels.core.config import (
        MLXParallelsConfig, ModelConfig, BatchConfig, EmbeddingConfig,
        get_embedding_config
    )
    BATCH_PROCESSING_AVAILABLE = True
except ImportError:
    BATCH_PROCESSING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingEngineConfig:
    """Konfiguration für Embedding Engine"""
    model_name: str = "mlx-community/gte-small"
    max_batch_size: int = 128
    embedding_dim: int = 384
    normalize: bool = True
    pooling_method: str = "mean"  # "mean", "max", "cls"
    cache_enabled: bool = True
    cache_size: int = 10000
    trust_remote_code: bool = True
    
    # Performance Settings
    enable_batch_processing: bool = True
    performance_mode: str = "balanced"  # "fast", "balanced", "throughput"
    chunk_size: int = 2000
    overlap: int = 50


class MLXEmbeddingEngine:
    """
    Enhanced MLX Embedding Engine mit Batch-Processing
    
    Features:
    - Batch-Embedding für dramatisch bessere Performance
    - Integration mit mlx-vector-db
    - Multi-User Support
    - Intelligent Caching
    - MLX-optimiert für Apple Silicon
    """
    
    def __init__(self, config: EmbeddingEngineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        # Caching
        self._cache = {} if config.cache_enabled else None
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Performance Tracking
        self.stats = {
            'total_embeddings': 0,
            'total_time': 0.0,
            'avg_embeddings_per_sec': 0.0,
            'cache_hit_rate': 0.0,
            'batch_requests': 0,
            'single_requests': 0
        }
        
        # MLX Parallels Integration
        if BATCH_PROCESSING_AVAILABLE and config.enable_batch_processing:
            try:
                mlx_config = get_embedding_config(config.model_name)
                # Override mit unseren Settings
                mlx_config.batch.max_batch_size = config.max_batch_size
                mlx_config.embedding.normalize = config.normalize
                mlx_config.embedding.pooling_method = config.pooling_method
                mlx_config.embedding.chunk_size = config.chunk_size
                
                self.batch_processor = BatchProcessor(mlx_config)
                self.batch_processing_enabled = True
                logger.info("✅ Embedding Batch-Processing aktiviert")
            except Exception as e:
                logger.warning(f"Batch-Processing Initialisierung fehlgeschlagen: {e}")
                self.batch_processor = None
                self.batch_processing_enabled = False
        else:
            self.batch_processor = None
            self.batch_processing_enabled = False
            
        logger.info(f"EmbeddingEngine initialisiert - Modell: {config.model_name}")
    
    async def initialize(self) -> bool:
        """Initialisiert Embedding Engine"""
        try:
            # MLX Parallels Batch Processor verwenden falls verfügbar
            if self.batch_processing_enabled and self.batch_processor:
                success = self.batch_processor.load_model()
                if success:
                    self.model_loaded = True
                    logger.info("🚀 Enhanced Embedding Engine mit Batch-Processing initialisiert")
                    return True
                else:
                    logger.warning("Batch Processor fehlgeschlagen - fallback zu Legacy")
            
            # Legacy-Initialisierung
            logger.info("🔄 Legacy Embedding Engine wird initialisiert...")
            
            self.model, self.tokenizer = load(
                self.config.model_name,
                tokenizer_config={
                    "trust_remote_code": self.config.trust_remote_code
                }
            )
            
            # Modell evaluieren
            if hasattr(self.model, 'parameters'):
                mx.eval(self.model.parameters())
            
            self.model_loaded = True
            logger.info("✅ Legacy Embedding Engine initialisiert")
            return True
            
        except Exception as e:
            logger.error(f"Embedding Engine Initialisierung fehlgeschlagen: {e}")
            return False
    
    async def encode_text(
        self,
        text: str,
        normalize: Optional[bool] = None,
        use_cache: bool = True
    ) -> List[float]:
        """
        Einzelnen Text zu Embedding konvertieren
        Optimiert durch interne Batch-Verarbeitung
        """
        return (await self.encode_texts([text], normalize, use_cache))[0]
    
    async def encode_texts(
        self,
        texts: List[str],
        normalize: Optional[bool] = None,
        use_cache: bool = True
    ) -> List[List[float]]:
        """
        Hauptmethode: Batch-Embedding für optimale Performance
        """
        if not texts:
            return []
        
        if not self.model_loaded:
            if not await self.initialize():
                raise RuntimeError("Embedding Engine konnte nicht initialisiert werden")
        
        normalize = normalize if normalize is not None else self.config.normalize
        
        # Cache-Check für einzelne Texte
        embeddings = []
        texts_to_process = []
        cache_indices = {}
        
        if use_cache and self._cache is not None:
            for i, text in enumerate(texts):
                cache_key = self._generate_cache_key(text, normalize)
                if cache_key in self._cache:
                    embeddings.append(self._cache[cache_key])
                    self._cache_hits += 1
                else:
                    cache_indices[len(texts_to_process)] = (i, cache_key)
                    texts_to_process.append(text)
                    embeddings.append(None)  # Placeholder
                    self._cache_misses += 1
        else:
            texts_to_process = texts
            embeddings = [None] * len(texts)
        
        # Batch-Processing für nicht-gecachte Texte
        if texts_to_process:
            start_time = time.time()
            
            try:
                if self.batch_processing_enabled and self.batch_processor:
                    # Enhanced Batch-Processing
                    result = await self.batch_processor.async_batch_embed(
                        texts=texts_to_process,
                        normalize=normalize
                    )
                    new_embeddings = result.embeddings
                    
                    self.stats['batch_requests'] += 1
                    logger.debug(f"Batch-Embedding: {len(texts_to_process)} Texte, "
                               f"{result.embeddings_per_second:.1f} embeddings/s")
                else:
                    # Legacy Sequential Processing
                    new_embeddings = []
                    for text in texts_to_process:
                        embedding = await self._encode_single_legacy(text, normalize)
                        new_embeddings.append(embedding)
                    
                    self.stats['single_requests'] += len(texts_to_process)
                
                # Ergebnisse in Original-Reihenfolge einfügen
                if use_cache and self._cache is not None:
                    for proc_idx, embedding in enumerate(new_embeddings):
                        orig_idx, cache_key = cache_indices[proc_idx]
                        embeddings[orig_idx] = embedding
                        
                        # Cache speichern
                        if len(self._cache) < self.config.cache_size:
                            self._cache[cache_key] = embedding
                        elif len(self._cache) >= self.config.cache_size:
                            # LRU: Entferne ältesten Eintrag
                            oldest_key = next(iter(self._cache))
                            del self._cache[oldest_key]
                            self._cache[cache_key] = embedding
                else:
                    embeddings = new_embeddings
                
                # Stats aktualisieren
                processing_time = time.time() - start_time
                self._update_stats(len(texts_to_process), processing_time)
                
            except Exception as e:
                logger.error(f"Batch-Embedding fehlgeschlagen: {e}")
                raise
        
        return embeddings
    
    async def _encode_single_legacy(
        self,
        text: str,
        normalize: bool
    ) -> List[float]:
        """Legacy Einzeltext-Encoding für Fallback"""
        try:
            # Text tokenisieren
            tokens = self.tokenizer.encode(text)
            if not isinstance(tokens, mx.array):
                tokens = mx.array(tokens)
            
            # Embedding extrahieren
            embedding = await self._extract_embedding_legacy(tokens)
            
            if normalize:
                # L2-Normalisierung
                norm = mx.linalg.norm(embedding)
                if norm.item() > 0:
                    embedding = embedding / norm
            
            # Zu Python Liste konvertieren
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            else:
                return [float(x) for x in embedding]
                
        except Exception as e:
            logger.warning(f"Legacy Encoding fehlgeschlagen: {e}")
            # Fallback: Null-Embedding
            return [0.0] * self.config.embedding_dim
    
    async def _extract_embedding_legacy(self, tokens: mx.array) -> mx.array:
        """Legacy Embedding-Extraktion"""
        try:
            # Forward pass
            if tokens.ndim == 1:
                tokens = tokens.reshape(1, -1)
            
            if hasattr(self.model, '__call__'):
                outputs = self.model(tokens)
            else:
                outputs = self.model.forward(tokens)
            
            # Pooling
            if self.config.pooling_method == "mean":
                embedding = mx.mean(outputs, axis=1)
            elif self.config.pooling_method == "max":
                embedding = mx.max(outputs, axis=1)
            elif self.config.pooling_method == "cls":
                embedding = outputs[:, 0]
            else:
                embedding = mx.mean(outputs, axis=1)
            
            return embedding.squeeze(0)
            
        except Exception as e:
            logger.debug(f"Legacy Embedding-Extraktion fehlgeschlagen: {e}")
            return mx.random.normal((self.config.embedding_dim,))
    
    # Integration mit mlx-vector-db
    
    async def embed_for_vector_db(
        self,
        texts: List[str],
        user_id: str,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Spezielle Embedding-Funktion für mlx-vector-db Integration
        """
        try:
            embeddings = await self.encode_texts(texts)
            
            # Metadata erweitern
            if metadata is None:
                metadata = []
            
            # Ensure metadata has same length as texts
            while len(metadata) < len(texts):
                metadata.append({})
            
            # Add user_id and timestamps to metadata
            for i, meta in enumerate(metadata):
                meta.update({
                    'user_id': user_id,
                    'text': texts[i],
                    'timestamp': time.time(),
                    'embedding_model': self.config.model_name,
                    'index': i
                })
            
            return {
                'embeddings': embeddings,
                'metadata': metadata,
                'count': len(embeddings),
                'model': self.config.model_name,
                'user_id': user_id
            }
            
        except Exception as e:
            logger.error(f"Vector DB Embedding fehlgeschlagen: {e}")
            raise
    
    async def embed_documents_chunked(
        self,
        documents: List[str],
        user_id: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Dokumenten-Chunking und Batch-Embedding für große Dokumente
        """
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.overlap
        
        all_chunks = []
        all_metadata = []
        
        for doc_idx, document in enumerate(documents):
            # Text in Chunks aufteilen
            chunks = self._chunk_text(document, chunk_size, overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'document_id': doc_idx,
                    'chunk_id': chunk_idx,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk),
                    'document_length': len(document)
                })
        
        # Batch-Embedding für alle Chunks
        result = await self.embed_for_vector_db(
            texts=all_chunks,
            user_id=user_id,
            metadata=all_metadata
        )
        
        return [{
            'embeddings': result['embeddings'],
            'metadata': result['metadata'],
            'chunks': all_chunks,
            'total_chunks': len(all_chunks),
            'total_documents': len(documents)
        }]
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Teilt Text in überlappende Chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Versuche an Wortgrenze zu brechen
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.7:  # Mindestens 70% der Chunk-Größe
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def _generate_cache_key(self, text: str, normalize: bool) -> str:
        """Generiert Cache-Key für Text"""
        key_components = [
            text,
            str(normalize),
            self.config.model_name,
            self.config.pooling_method
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_stats(self, count: int, processing_time: float):
        """Aktualisiert Performance-Statistiken"""
        self.stats['total_embeddings'] += count
        self.stats['total_time'] += processing_time
        
        if processing_time > 0:
            current_eps = count / processing_time
            self.stats['avg_embeddings_per_sec'] = (
                self.stats['avg_embeddings_per_sec'] * 0.9 + current_eps * 0.1
            )
        
        # Cache Hit Rate
        total_requests = self._cache_hits + self._cache_misses
        if total_requests > 0:
            self.stats['cache_hit_rate'] = self._cache_hits / total_requests
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt Performance-Statistiken zurück"""
        stats = {
            **self.stats,
            'model_loaded': self.model_loaded,
            'model_name': self.config.model_name,
            'batch_processing_enabled': self.batch_processing_enabled,
            'cache_enabled': self._cache is not None,
            'cache_size': len(self._cache) if self._cache else 0,
            'config': {
                'max_batch_size': self.config.max_batch_size,
                'embedding_dim': self.config.embedding_dim,
                'normalize': self.config.normalize,
                'pooling_method': self.config.pooling_method
            }
        }
        
        if self.batch_processing_enabled and self.batch_processor:
            try:
                batch_stats = self.batch_processor.get_stats()
                stats['batch_processor_stats'] = batch_stats
            except Exception as e:
                logger.debug(f"Batch Processor Stats fehlgeschlagen: {e}")
        
        return stats
    
    def clear_cache(self):
        """Leert den Embedding-Cache"""
        if self._cache is not None:
            self._cache.clear()
            self._cache_hits = 0
            self._cache_misses = 0
            logger.info("Embedding Cache geleert")
    
    async def cleanup(self):
        """Cleanup-Methode"""
        if self.batch_processor:
            self.batch_processor.cleanup()
        
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        
        if self._cache:
            self._cache.clear()


# Utility Functions

async def create_embedding_engine(
    model_name: str = "mlx-community/gte-small",
    enable_batch_processing: bool = True
) -> MLXEmbeddingEngine:
    """Factory Function für Embedding Engine"""
    config = EmbeddingEngineConfig(
        model_name=model_name,
        enable_batch_processing=enable_batch_processing
    )
    
    engine = MLXEmbeddingEngine(config)
    await engine.initialize()
    return engine


def get_compatible_embedding_models() -> List[str]:
    """Gibt Liste kompatibler Embedding-Modelle zurück"""
    return [
        "mlx-community/gte-small",
        "mlx-community/gte-large",
        "mlx-community/all-MiniLM-L6-v2", 
        "mlx-community/bge-small-en",
        "mlx-community/bge-large-en",
        "mlx-community/e5-small-v2",
        "mlx-community/e5-large-v2"
    ]


async def benchmark_embedding_performance(
    model_name: str,
    test_texts: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Benchmark für Embedding-Performance"""
    if test_texts is None:
        test_texts = [
            "MLX ist ein Framework für Apple Silicon",
            "Unified Memory bietet Performance-Vorteile", 
            "Neural Engine beschleunigt ML-Workloads",
            "Metal Performance Shaders für GPU-Computing",
            "ARM-Prozessoren sind energieeffizient"
        ] * 20  # 100 Test-Texte
    
    engine = await create_embedding_engine(model_name)
    
    # Single Processing Benchmark
    start_time = time.time()
    single_results = []
    for text in test_texts:
        embedding = await engine.encode_text(text, use_cache=False)
        single_results.append(embedding)
    single_time = time.time() - start_time
    
    # Batch Processing Benchmark  
    start_time = time.time()
    batch_results = await engine.encode_texts(test_texts, use_cache=False)
    batch_time = time.time() - start_time
    
    await engine.cleanup()
    
    return {
        'model_name': model_name,
        'test_count': len(test_texts),
        'single_time': single_time,
        'batch_time': batch_time,
        'speedup': single_time / batch_time if batch_time > 0 else 0,
        'single_eps': len(test_texts) / single_time,
        'batch_eps': len(test_texts) / batch_time,
        'embedding_dim': len(batch_results[0]) if batch_results else 0
    }