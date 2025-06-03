"""
MLX Embedding Engine
Optimiert für Apple Silicon mit MLX Framework
Unterstützt verschiedene Embedding-Modelle und Batch-Processing
"""

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass
import json
import hashlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
import numpy as np

@dataclass
class EmbeddingConfig:
    """Konfiguration für Embedding Engine"""
    model_path: str = "mlx-community/gte-small"
    max_sequence_length: int = 512
    batch_size: int = 32
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    cache_size: int = 1000
    device: str = "auto"  # auto, cpu, gpu

@dataclass
class EmbeddingResult:
    """Result-Container für Embeddings"""
    embeddings: mx.array
    texts: List[str]
    model_name: str
    processing_time: float
    cached_count: int = 0

class MLXEmbeddingEngine:
    """
    High-Performance Embedding Engine für MLX
    
    Features:
    - Multiple Embedding Models Support
    - Intelligent Caching System
    - Batch Processing Optimization
    - Memory-Efficient Operations
    - Apple Silicon Optimization
    """
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.embedding_dim = None
        
        # Caching System
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance Metrics
        self.total_embeddings = 0
        self.total_processing_time = 0.0
        
        # Supported Models
        self.supported_models = {
            "gte-small": "mlx-community/gte-small",
            "gte-large": "mlx-community/gte-large", 
            "all-MiniLM-L6-v2": "mlx-community/all-MiniLM-L6-v2",
            "bge-small-en": "mlx-community/bge-small-en",
            "bge-large-en": "mlx-community/bge-large-en",
            "e5-small-v2": "mlx-community/e5-small-v2",
            "e5-large-v2": "mlx-community/e5-large-v2"
        }
    
    async def initialize(self, model_path: Optional[str] = None) -> None:
        """
        Lazy Model Loading für Memory Efficiency
        """
        target_model = model_path or self.config.model_path
        
        if self.model is None or self.model_name != target_model:
            print(f"Loading embedding model: {target_model}")
            start_time = time.time()
            
            try:
                # Load model using MLX
                self.model, self.tokenizer = load(target_model)
                
                # Ensure model is evaluated and cached in memory
                mx.eval(self.model.parameters())
                
                self.model_name = target_model
                
                # Determine embedding dimension
                self.embedding_dim = await self._get_embedding_dimension()
                
                load_time = time.time() - start_time
                print(f"✅ Model loaded in {load_time:.2f}s, embedding dim: {self.embedding_dim}")
                
            except Exception as e:
                print(f"❌ Error loading model {target_model}: {e}")
                raise
    
    async def embed(self, 
                   texts: Union[str, List[str]], 
                   model_path: Optional[str] = None,
                   normalize: Optional[bool] = None) -> EmbeddingResult:
        """
        Hauptfunktion: Erstellt Embeddings für Text(e)
        
        Args:
            texts: Einzelner Text oder Liste von Texten
            model_path: Optional anderes Modell verwenden
            normalize: Override für Normalisierung
            
        Returns:
            EmbeddingResult mit Embeddings und Metadaten
        """
        start_time = time.time()
        
        # Ensure model is loaded
        await self.initialize(model_path)
        
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return EmbeddingResult(
                embeddings=mx.array([]),
                texts=[],
                model_name=self.model_name,
                processing_time=0.0
            )
        
        # Check cache for existing embeddings
        cached_embeddings, non_cached_texts, cache_indices = self._check_cache(texts)
        
        # Process non-cached texts
        if non_cached_texts:
            new_embeddings = await self._compute_embeddings(non_cached_texts)
            
            # Update cache
            self._update_cache(non_cached_texts, new_embeddings)
        else:
            new_embeddings = mx.array([])
        
        # Combine cached and new embeddings
        final_embeddings = self._combine_embeddings(
            cached_embeddings, new_embeddings, cache_indices, texts
        )
        
        # Normalize if requested
        if normalize if normalize is not None else self.config.normalize_embeddings:
            final_embeddings = self._normalize_embeddings(final_embeddings)
        
        processing_time = time.time() - start_time
        
        # Update metrics
        self.total_embeddings += len(texts)
        self.total_processing_time += processing_time
        
        return EmbeddingResult(
            embeddings=final_embeddings,
            texts=texts,
            model_name=self.model_name,
            processing_time=processing_time,
            cached_count=len(texts) - len(non_cached_texts)
        )
    
    async def embed_batch(self, 
                         text_batches: List[List[str]], 
                         model_path: Optional[str] = None) -> List[EmbeddingResult]:
        """
        Batch Processing für große Mengen von Texten
        """
        await self.initialize(model_path)
        
        results = []
        for batch in text_batches:
            result = await self.embed(batch, model_path)
            results.append(result)
        
        return results
    
    async def _compute_embeddings(self, texts: List[str]) -> mx.array:
        """
        Core Embedding Computation mit MLX
        """
        try:
            # Tokenize texts
            inputs = []
            for text in texts:
                # Preprocessing für bessere Embeddings
                processed_text = self._preprocess_text(text)
                
                # Tokenize with proper handling
                tokens = self.tokenizer.encode(processed_text)
                
                # Truncate if necessary
                if len(tokens) > self.config.max_sequence_length:
                    tokens = tokens[:self.config.max_sequence_length]
                
                inputs.append(tokens)
            
            # Batch processing
            embeddings = []
            batch_size = self.config.batch_size
            
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                batch_embeddings = await self._process_batch(batch_inputs)
                embeddings.extend(batch_embeddings)
            
            return mx.array(embeddings)
            
        except Exception as e:
            print(f"Error computing embeddings: {e}")
            # Fallback: return zero embeddings
            return mx.zeros((len(texts), self.embedding_dim or 384))
    
    async def _process_batch(self, batch_inputs: List[List[int]]) -> List[List[float]]:
        """
        Verarbeitet einen Batch von tokenisierten Inputs
        """
        try:
            # Pad batch to same length
            max_length = max(len(tokens) for tokens in batch_inputs)
            padded_inputs = []
            
            for tokens in batch_inputs:
                padded = tokens + [0] * (max_length - len(tokens))
                padded_inputs.append(padded)
            
            # Convert to MLX array
            input_ids = mx.array(padded_inputs)
            
            # Forward pass through model
            with mx.no_grad():
                outputs = self.model(input_ids)
                
                # Extract embeddings (meist letzter hidden state)
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states'):
                    embeddings = outputs.hidden_states[-1]
                else:
                    embeddings = outputs
                
                # Mean pooling
                embeddings = mx.mean(embeddings, axis=1)
                
                # Ensure evaluation
                mx.eval(embeddings)
                
                return embeddings.tolist()
                
        except Exception as e:
            print(f"Error in batch processing: {e}")
            # Fallback
            return [[0.0] * (self.embedding_dim or 384) for _ in batch_inputs]
    
    def _preprocess_text(self, text: str) -> str:
        """
        Text Preprocessing für bessere Embedding-Qualität
        """
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Handle empty text
        if not text:
            text = "[EMPTY]"
        
        return text
    
    def _check_cache(self, texts: List[str]) -> Tuple[Dict[int, mx.array], List[str], Dict[str, int]]:
        """
        Prüft Cache für existierende Embeddings
        """
        if not self.config.cache_embeddings:
            return {}, texts, {}
        
        cached_embeddings = {}
        non_cached_texts = []
        cache_indices = {}
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            
            if cache_key in self.cache:
                cached_embeddings[i] = self.cache[cache_key]
                self.cache_hits += 1
            else:
                cache_indices[text] = len(non_cached_texts)
                non_cached_texts.append(text)
                self.cache_misses += 1
        
        return cached_embeddings, non_cached_texts, cache_indices
    
    def _update_cache(self, texts: List[str], embeddings: mx.array) -> None:
        """
        Updated Cache mit neuen Embeddings
        """
        if not self.config.cache_embeddings:
            return
        
        # Cache size management
        if len(self.cache) > self.config.cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.cache.keys())[:len(self.cache) - self.config.cache_size + len(texts)]
            for key in keys_to_remove:
                del self.cache[key]
        
        # Add new embeddings to cache
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            self.cache[cache_key] = embeddings[i]
    
    def _combine_embeddings(self, 
                           cached_embeddings: Dict[int, mx.array], 
                           new_embeddings: mx.array, 
                           cache_indices: Dict[str, int], 
                           original_texts: List[str]) -> mx.array:
        """
        Kombiniert gecachte und neue Embeddings in richtiger Reihenfolge
        """
        result_embeddings = []
        new_embedding_idx = 0
        
        for i, text in enumerate(original_texts):
            if i in cached_embeddings:
                result_embeddings.append(cached_embeddings[i])
            else:
                if new_embedding_idx < len(new_embeddings):
                    result_embeddings.append(new_embeddings[new_embedding_idx])
                    new_embedding_idx += 1
                else:
                    # Fallback: zero embedding
                    result_embeddings.append(mx.zeros(self.embedding_dim or 384))
        
        return mx.stack(result_embeddings)
    
    def _normalize_embeddings(self, embeddings: mx.array) -> mx.array:
        """
        L2-Normalisierung der Embeddings
        """
        norms = mx.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = mx.maximum(norms, 1e-8)  # Avoid division by zero
        return embeddings / norms
    
    def _get_cache_key(self, text: str) -> str:
        """
        Erstellt Cache-Key für Text
        """
        content = f"{self.model_name}:{text}:{self.config.max_sequence_length}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _get_embedding_dimension(self) -> int:
        """
        Bestimmt Embedding-Dimension des geladenen Modells
        """
        try:
            # Test with dummy input
            test_embedding = await self._compute_embeddings(["test"])
            return test_embedding.shape[1]
        except:
            # Common dimensions for different models
            model_dims = {
                "gte-small": 384,
                "gte-large": 1024,
                "all-MiniLM-L6-v2": 384,
                "bge-small-en": 384,
                "bge-large-en": 1024,
                "e5-small-v2": 384,
                "e5-large-v2": 1024
            }
            
            for model_key, dim in model_dims.items():
                if model_key in self.model_name:
                    return dim
            
            return 384  # Default fallback
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Liefert Performance-Statistiken
        """
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        avg_processing_time = self.total_processing_time / self.total_embeddings if self.total_embeddings > 0 else 0
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "total_embeddings": self.total_embeddings,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "supported_models": list(self.supported_models.keys())
        }
    
    def clear_cache(self) -> None:
        """
        Leert den Embedding-Cache
        """
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    async def benchmark(self, texts: List[str] = None) -> Dict[str, float]:
        """
        Performance Benchmark für verschiedene Szenarien
        """
        if texts is None:
            texts = [
                "This is a simple test sentence.",
                "Here's another sentence for testing the embedding engine.",
                "A longer sentence with more content to test the processing capabilities of our MLX-based embedding system.",
                "Short text.",
                "Medium length text with some technical terms like machine learning, neural networks, and natural language processing.",
            ] * 10  # 50 texts total
        
        print(f"Running benchmark with {len(texts)} texts...")
        
        # Warm up
        await self.embed(texts[:5])
        
        # Clear cache for fair comparison
        self.clear_cache()
        
        # Test without cache
        start_time = time.time()
        result_no_cache = await self.embed(texts)
        time_no_cache = time.time() - start_time
        
        # Test with cache (second run)
        start_time = time.time()
        result_with_cache = await self.embed(texts)
        time_with_cache = time.time() - start_time
        
        return {
            "texts_count": len(texts),
            "time_no_cache": time_no_cache,
            "time_with_cache": time_with_cache,
            "speedup_factor": time_no_cache / time_with_cache if time_with_cache > 0 else 0,
            "embeddings_per_second_no_cache": len(texts) / time_no_cache,
            "embeddings_per_second_with_cache": len(texts) / time_with_cache,
            "cache_hit_rate": result_with_cache.cached_count / len(texts)
        }

# Usage Examples
async def example_usage():
    """Beispiele für die Nutzung der Embedding Engine"""
    
    # Initialize engine
    config = EmbeddingConfig(
        model_path="mlx-community/gte-small",
        batch_size=16,
        cache_embeddings=True
    )
    
    engine = MLXEmbeddingEngine(config)
    
    # Single text embedding
    result = await engine.embed("This is a test sentence.")
    print(f"Single embedding shape: {result.embeddings.shape}")
    
    # Multiple text embeddings
    texts = [
        "First document about machine learning.",
        "Second document about neural networks.", 
        "Third document about natural language processing."
    ]
    
    result = await engine.embed(texts)
    print(f"Batch embedding shape: {result.embeddings.shape}")
    print(f"Processing time: {result.processing_time:.3f}s")
    print(f"Cached: {result.cached_count}/{len(texts)}")
    
    # Performance stats
    stats = engine.get_stats()
    print(f"Performance stats: {stats}")
    
    # Benchmark
    benchmark_results = await engine.benchmark()
    print(f"Benchmark results: {benchmark_results}")

if __name__ == "__main__":
    asyncio.run(example_usage())