#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

"""
MLX Vector Store - Enhanced Integration mit mlx-vector-db
Batch-Operationen und Multi-User Support für optimale Performance
"""

import asyncio
import aiohttp
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Konfiguration für Vector Store"""
    base_url: str = "http://localhost:8000"
    api_key: str = "your-secure-key"
    default_user_id: str = "default_user"
    default_model_id: str = "gte-small"
    
    # Batch-Processing Settings
    max_batch_size: int = 100
    batch_timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Performance Settings
    connection_timeout: float = 10.0
    read_timeout: float = 30.0
    max_connections: int = 10


@dataclass
class SearchResult:
    """Einzelnes Suchergebnis"""
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    document_id: Optional[str] = None


@dataclass
class BatchSearchResult:
    """Batch-Suchergebnis"""
    results: List[List[SearchResult]]
    processing_time: float
    total_queries: int
    avg_results_per_query: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLXVectorStore:
    """
    Enhanced MLX Vector Store mit Batch-Processing
    
    Features:
    - Batch-Add für dramatisch bessere Performance beim Indexieren
    - Batch-Query für parallele Suche
    - Multi-User Support mit User-Isolation  
    - Optimierte Integration mit mlx-vector-db
    - Robuste Fehlerbehandlung und Retry-Logic
    """
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.session = None
        
        # Performance Tracking
        self.stats = {
            'total_adds': 0,
            'total_queries': 0,
            'total_batch_adds': 0,
            'total_batch_queries': 0,
            'avg_add_time': 0.0,
            'avg_query_time': 0.0,
            'connection_errors': 0,
            'retry_count': 0
        }
        
        logger.info(f"VectorStore initialisiert - Base URL: {config.base_url}")
    
    async def __aenter__(self):
        """Async Context Manager Entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async Context Manager Exit"""
        await self.cleanup()
    
    async def initialize(self):
        """Initialisiert HTTP Session"""
        timeout = aiohttp.ClientTimeout(
            connect=self.config.connection_timeout,
            total=self.config.read_timeout
        )
        
        connector = aiohttp.TCPConnector(
            limit=self.config.max_connections,
            limit_per_host=self.config.max_connections
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "X-API-Key": self.config.api_key,
                "Content-Type": "application/json"
            }
        )
        
        # Health Check
        try:
            await self._health_check()
            logger.info("✅ Vector Store Verbindung erfolgreich")
        except Exception as e:
            logger.warning(f"⚠️  Health Check fehlgeschlagen: {e}")
    
    async def _health_check(self) -> bool:
        """Überprüft Vector Store Verfügbarkeit"""
        try:
            async with self.session.get(f"{self.config.base_url}/monitoring/health") as response:
                if response.status == 200:
                    return True
                else:
                    logger.warning(f"Health Check Status: {response.status}")
                    return False
        except Exception as e:
            logger.debug(f"Health Check Fehler: {e}")
            return False
    
    # User/Store Management
    
    async def ensure_store_exists(
        self,
        user_id: str,
        model_id: str
    ) -> bool:
        """Stellt sicher dass Store für User/Model existiert"""
        try:
            payload = {
                "user_id": user_id,
                "model_id": model_id
            }
            
            async with self.session.post(
                f"{self.config.base_url}/admin/create_store",
                json=payload
            ) as response:
                if response.status in [200, 201, 409]:  # 409 = Already exists
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Store Creation fehlgeschlagen: {response.status} - {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Store Creation Fehler: {e}")
            return False
    
    # Batch-Add Operations
    
    async def add_vectors_batch(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> bool:
        """
        Batch-Add für optimale Performance beim Indexieren
        """
        user_id = user_id or self.config.default_user_id
        model_id = model_id or self.config.default_model_id
        
        if len(embeddings) != len(metadata):
            raise ValueError("Anzahl Embeddings muss Anzahl Metadata entsprechen")
        
        start_time = time.time()
        
        try:
            # Store sicherstellen
            await self.ensure_store_exists(user_id, model_id)
            
            # Große Batches aufteilen
            success_count = 0
            total_batches = 0
            
            for i in range(0, len(embeddings), self.config.max_batch_size):
                batch_embeddings = embeddings[i:i + self.config.max_batch_size]
                batch_metadata = metadata[i:i + self.config.max_batch_size]
                
                success = await self._add_batch_chunk(
                    batch_embeddings, batch_metadata, user_id, model_id
                )
                
                if success:
                    success_count += 1
                total_batches += 1
                
                logger.debug(f"Batch {total_batches} von {(len(embeddings) + self.config.max_batch_size - 1) // self.config.max_batch_size} verarbeitet")
            
            processing_time = time.time() - start_time
            
            # Stats aktualisieren
            self.stats['total_batch_adds'] += 1
            self.stats['total_adds'] += len(embeddings)
            self._update_avg_time('add', processing_time)
            
            success_rate = success_count / total_batches if total_batches > 0 else 0
            logger.info(f"Batch-Add abgeschlossen: {len(embeddings)} Vektoren, "
                       f"{success_rate:.1%} Erfolgsrate, {processing_time:.2f}s")
            
            return success_rate > 0.8  # 80% Erfolgsrate als Mindestanforderung
            
        except Exception as e:
            logger.error(f"Batch-Add fehlgeschlagen: {e}")
            self.stats['connection_errors'] += 1
            return False
    
    async def _add_batch_chunk(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        user_id: str,
        model_id: str
    ) -> bool:
        """Fügt einzelnen Batch-Chunk hinzu mit Retry-Logic"""
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "vectors": embeddings,
            "metadata": metadata
        }
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.post(
                    f"{self.config.base_url}/vectors/add",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.batch_timeout)
                ) as response:
                    
                    if response.status == 200:
                        return True
                    else:
                        error_text = await response.text()
                        logger.warning(f"Add Batch Chunk fehlgeschlagen (Versuch {attempt + 1}): "
                                     f"{response.status} - {error_text}")
                        
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
                        
            except asyncio.TimeoutError:
                logger.warning(f"Add Batch Chunk Timeout (Versuch {attempt + 1})")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            except Exception as e:
                logger.warning(f"Add Batch Chunk Fehler (Versuch {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
        
        self.stats['retry_count'] += self.config.max_retries
        return False
    
    # Single Add (Legacy-kompatibel)
    
    async def add_vector(
        self,
        embedding: List[float],
        metadata: Dict[str, Any],
        user_id: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> bool:
        """Einzelnen Vektor hinzufügen (Legacy-kompatibel)"""
        return await self.add_vectors_batch([embedding], [metadata], user_id, model_id)
    
    # Batch-Query Operations
    
    async def batch_similarity_search(
        self,
        query_embeddings: List[List[float]],
        k: int = 5,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> BatchSearchResult:
        """
        Batch-Similarity-Search für parallele Abfragen
        """
        user_id = user_id or self.config.default_user_id
        model_id = model_id or self.config.default_model_id
        
        start_time = time.time()
        all_results = []
        
        try:
            # Parallele Queries ausführen
            tasks = []
            for embedding in query_embeddings:
                task = self._single_similarity_search(
                    embedding, k, user_id, model_id, filters
                )
                tasks.append(task)
            
            # Alle Queries parallel ausführen
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Ergebnisse verarbeiten
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Query {i} fehlgeschlagen: {result}")
                    all_results.append([])
                else:
                    all_results.append(result)
            
            processing_time = time.time() - start_time
            
            # Stats aktualisieren
            self.stats['total_batch_queries'] += 1
            self.stats['total_queries'] += len(query_embeddings)
            self._update_avg_time('query', processing_time)
            
            # Ergebnis-Statistiken
            total_results = sum(len(results) for results in all_results)
            avg_results = total_results / len(query_embeddings) if query_embeddings else 0
            
            logger.info(f"Batch-Query abgeschlossen: {len(query_embeddings)} Queries, "
                       f"{avg_results:.1f} avg results, {processing_time:.2f}s")
            
            return BatchSearchResult(
                results=all_results,
                processing_time=processing_time,
                total_queries=len(query_embeddings),
                avg_results_per_query=avg_results,
                metadata={
                    'user_id': user_id,
                    'model_id': model_id,
                    'k': k,
                    'filters': filters
                }
            )
            
        except Exception as e:
            logger.error(f"Batch-Query fehlgeschlagen: {e}")
            self.stats['connection_errors'] += 1
            
            # Fallback: Leere Ergebnisse
            return BatchSearchResult(
                results=[[] for _ in query_embeddings],
                processing_time=time.time() - start_time,
                total_queries=len(query_embeddings),
                avg_results_per_query=0,
                metadata={'error': str(e)}
            )
    
    async def _single_similarity_search(
        self,
        query_embedding: List[float],
        k: int,
        user_id: str,
        model_id: str,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """Einzelne Similarity Search"""
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "query": query_embedding,
            "k": k
        }
        
        if filters:
            payload["filters"] = filters
        
        try:
            async with self.session.post(
                f"{self.config.base_url}/vectors/query",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return self._parse_search_results(data)
                else:
                    error_text = await response.text()
                    logger.warning(f"Similarity Search fehlgeschlagen: {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.debug(f"Similarity Search Fehler: {e}")
            return []
    
    def _parse_search_results(self, data: List[Dict[str, Any]]) -> List[SearchResult]:
        """Konvertiert API-Antwort zu SearchResult-Objekten"""
        results = []
        
        for item in data:
            try:
                # Flexibles Parsing für verschiedene API-Formate
                content = ""
                score = 0.0
                metadata = {}
                
                if isinstance(item, dict):
                    # Score extrahieren
                    score = item.get('score', item.get('similarity', item.get('distance', 0.0)))
                    
                    # Content extrahieren
                    meta = item.get('metadata', {})
                    content = (
                        meta.get('text') or 
                        meta.get('content') or 
                        item.get('content') or 
                        item.get('text') or 
                        str(item)
                    )
                    
                    # Metadata verarbeiten
                    metadata = meta.copy() if isinstance(meta, dict) else {}
                    metadata.update({
                        'user_id': metadata.get('user_id'),
                        'document_id': metadata.get('document_id') or metadata.get('id'),
                        'index': metadata.get('index')
                    })
                
                result = SearchResult(
                    content=content,
                    score=float(score),
                    metadata=metadata,
                    user_id=metadata.get('user_id'),
                    document_id=metadata.get('document_id')
                )
                
                results.append(result)
                
            except Exception as e:
                logger.debug(f"Result Parsing Fehler: {e}")
                continue
        
        return results
    
    # Legacy-kompatible Methoden
    
    async def similarity_search(
        self,
        query: Union[str, List[float]],
        k: int = 5,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Legacy-kompatible Similarity Search
        Unterstützt sowohl Text-Queries (benötigt Embedding Engine) als auch direkte Embeddings
        """
        
        # Falls Text-Query, muss zu Embedding konvertiert werden
        if isinstance(query, str):
            raise ValueError(
                "Text-Queries benötigen Embedding Engine. "
                "Verwenden Sie search_by_text() oder konvertieren Sie zu Embedding."
            )
        
        # Embedding-Query
        batch_result = await self.batch_similarity_search(
            query_embeddings=[query],
            k=k,
            user_id=user_id,
            model_id=model_id,
            filters=filters
        )
        
        return batch_result.results[0] if batch_result.results else []
    
    async def search_by_text(
        self,
        query_text: str,
        embedding_engine,
        k: int = 5,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Text-basierte Suche mit Embedding Engine
        """
        try:
            # Text zu Embedding konvertieren
            query_embedding = await embedding_engine.encode_text(query_text)
            
            # Similarity Search durchführen
            return await self.similarity_search(
                query=query_embedding,
                k=k,
                user_id=user_id,
                model_id=model_id,
                filters=filters
            )
            
        except Exception as e:
            logger.error(f"Text-basierte Suche fehlgeschlagen: {e}")
            return []
    
    async def batch_search_by_text(
        self,
        query_texts: List[str],
        embedding_engine,
        k: int = 5,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> BatchSearchResult:
        """
        Batch Text-basierte Suche mit Embedding Engine
        Optimal für RAG-Batch-Processing
        """
        try:
            # Batch-Embedding für alle Query-Texte
            query_embeddings = await embedding_engine.encode_texts(query_texts)
            
            # Batch Similarity Search
            return await self.batch_similarity_search(
                query_embeddings=query_embeddings,
                k=k,
                user_id=user_id,
                model_id=model_id,
                filters=filters
            )
            
        except Exception as e:
            logger.error(f"Batch Text-Suche fehlgeschlagen: {e}")
            return BatchSearchResult(
                results=[[] for _ in query_texts],
                processing_time=0.0,
                total_queries=len(query_texts),
                avg_results_per_query=0,
                metadata={'error': str(e)}
            )
    
    # Store Management
    
    async def get_store_stats(
        self,
        user_id: Optional[str] = None,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Gibt Store-Statistiken zurück"""
        user_id = user_id or self.config.default_user_id
        model_id = model_id or self.config.default_model_id
        
        try:
            params = {
                "user_id": user_id,
                "model_id": model_id
            }
            
            async with self.session.get(
                f"{self.config.base_url}/vectors/count",
                params=params
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"Status {response.status}"}
                    
        except Exception as e:
            logger.debug(f"Store Stats Fehler: {e}")
            return {"error": str(e)}
    
    async def delete_store(
        self,
        user_id: str,
        model_id: str,
        confirm: bool = False
    ) -> bool:
        """Löscht kompletten Store (Vorsicht!)"""
        if not confirm:
            logger.warning("Store-Löschung benötigt confirm=True")
            return False
        
        try:
            payload = {
                "user_id": user_id,
                "model_id": model_id,
                "confirm": True
            }
            
            async with self.session.delete(
                f"{self.config.base_url}/admin/delete_store",
                json=payload
            ) as response:
                
                return response.status == 200
                
        except Exception as e:
            logger.error(f"Store-Löschung fehlgeschlagen: {e}")
            return False
    
    # Performance & Monitoring
    
    def _update_avg_time(self, operation: str, time_taken: float):
        """Aktualisiert durchschnittliche Ausführungszeiten"""
        key = f'avg_{operation}_time'
        if key in self.stats:
            self.stats[key] = self.stats[key] * 0.9 + time_taken * 0.1
        else:
            self.stats[key] = time_taken
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt Performance-Statistiken zurück"""
        return {
            **self.stats,
            'config': {
                'base_url': self.config.base_url,
                'max_batch_size': self.config.max_batch_size,
                'batch_timeout': self.config.batch_timeout,
                'max_retries': self.config.max_retries
            },
            'health': {
                'session_active': self.session is not None,
                'error_rate': self.stats['connection_errors'] / max(1, self.stats['total_queries'])
            }
        }
    
    async def optimize_performance(self):
        """Performance-Optimierung basierend auf aktuellen Stats"""
        try:
            # Adaptive Batch-Größe basierend auf Erfolgsrate
            error_rate = self.stats['connection_errors'] / max(1, self.stats['total_queries'])
            
            if error_rate > 0.1:  # Mehr als 10% Fehler
                self.config.max_batch_size = max(10, self.config.max_batch_size // 2)
                logger.info(f"Batch-Größe reduziert auf {self.config.max_batch_size} aufgrund hoher Fehlerrate")
            elif error_rate < 0.01:  # Weniger als 1% Fehler
                self.config.max_batch_size = min(200, self.config.max_batch_size * 1.2)
                logger.info(f"Batch-Größe erhöht auf {self.config.max_batch_size}")
            
        except Exception as e:
            logger.debug(f"Performance-Optimierung fehlgeschlagen: {e}")
    
    async def cleanup(self):
        """Cleanup-Methode"""
        if self.session:
            await self.session.close()
            self.session = None
        
        logger.info("Vector Store Cleanup abgeschlossen")


# Utility Functions

async def create_vector_store(
    base_url: str = "http://localhost:8000",
    api_key: str = "your-secure-key"
) -> MLXVectorStore:
    """Factory Function für Vector Store"""
    config = VectorStoreConfig(
        base_url=base_url,
        api_key=api_key
    )
    
    store = MLXVectorStore(config)
    await store.initialize()
    return store


async def test_vector_store_connection(
    base_url: str = "http://localhost:8000",
    api_key: str = "your-secure-key"
) -> Dict[str, Any]:
    """Testet Vector Store Verbindung"""
    try:
        async with create_vector_store(base_url, api_key) as store:
            # Health Check
            health = await store._health_check()
            
            # Store Stats
            stats = await store.get_store_stats()
            
            return {
                'connection': 'success',
                'health': health,
                'stats': stats,
                'base_url': base_url
            }
            
    except Exception as e:
        return {
            'connection': 'failed',
            'error': str(e),
            'base_url': base_url
        }


async def benchmark_vector_store_performance(
    store: MLXVectorStore,
    test_embeddings: Optional[List[List[float]]] = None,
    k: int = 5
) -> Dict[str, Any]:
    """Benchmark für Vector Store Performance"""
    
    if test_embeddings is None:
        # Generiere Test-Embeddings
        import random
        test_embeddings = [
            [random.random() for _ in range(384)]
            for _ in range(100)
        ]
    
    test_metadata = [
        {"text": f"Test document {i}", "id": i}
        for i in range(len(test_embeddings))
    ]
    
    # Add Benchmark
    start_time = time.time()
    add_success = await store.add_vectors_batch(
        embeddings=test_embeddings,
        metadata=test_metadata,
        user_id="benchmark_user"
    )
    add_time = time.time() - start_time
    
    # Query Benchmark
    query_embeddings = test_embeddings[:10]  # Erste 10 als Queries
    
    start_time = time.time()
    batch_result = await store.batch_similarity_search(
        query_embeddings=query_embeddings,
        k=k,
        user_id="benchmark_user"
    )
    query_time = time.time() - start_time
    
    return {
        'add_performance': {
            'success': add_success,
            'embeddings_count': len(test_embeddings),
            'time_seconds': add_time,
            'embeddings_per_second': len(test_embeddings) / add_time if add_time > 0 else 0
        },
        'query_performance': {
            'queries_count': len(query_embeddings),
            'time_seconds': query_time,
            'queries_per_second': len(query_embeddings) / query_time if query_time > 0 else 0,
            'avg_results_per_query': batch_result.avg_results_per_query,
            'total_results': sum(len(results) for results in batch_result.results)
        },
        'store_stats': store.get_performance_stats()
    }