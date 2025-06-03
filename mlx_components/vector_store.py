"""
MLX Vector Store
Integration mit mlx-vector-db für High-Performance Vector Operations
Optimiert für Multi-User Scenarios und Apple Silicon
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

import mlx.core as mx
import numpy as np

@dataclass
class VectorStoreConfig:
    """Konfiguration für Vector Store"""
    base_url: str = "http://localhost:8000"
    api_key: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    default_k: int = 5
    max_k: int = 100

@dataclass
class VectorDocument:
    """Dokument für Vector Store"""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    namespace: Optional[str] = None
    timestamp: Optional[str] = None

@dataclass
class QueryResult:
    """Result für Vector Queries"""
    id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None

@dataclass
class VectorStoreStats:
    """Statistiken für Vector Store Performance"""
    total_vectors: int
    query_latency_ms: float
    add_latency_ms: float
    storage_size_mb: float
    last_updated: str

class MLXVectorStore:
    """
    High-Performance Vector Store mit mlx-vector-db Integration
    
    Features:
    - Multi-User Support mit User-Isolation
    - High-Performance Queries (1000+ QPS target)
    - Batch Operations für Effizienz
    - Metadata Filtering
    - Connection Pooling
    - Automatic Retry Logic
    - Performance Monitoring
    """
    
    def __init__(self, config: VectorStoreConfig = None):
        self.config = config or VectorStoreConfig()
        self.session: Optional[aiohttp.ClientSession] = None
        self.stores: Dict[str, Dict[str, bool]] = {}  # Cache für User-Store Status
        
        # Performance Metrics
        self.query_count = 0
        self.add_count = 0
        self.total_query_time = 0.0
        self.total_add_time = 0.0
        
        # Headers für API Requests
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.config.api_key:
            self.headers["X-API-Key"] = self.config.api_key
    
    async def initialize(self) -> None:
        """
        Initialisiert HTTP Session und Connection Pool
        """
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            connector = aiohttp.TCPConnector(
                limit=100,  # Connection pool size
                limit_per_host=20,
                keepalive_timeout=60
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=self.headers
            )
            
            # Health check
            await self._health_check()
    
    async def close(self) -> None:
        """
        Schließt HTTP Session
        """
        if self.session:
            await self.session.close()
            self.session = None
    
    async def create_user_store(self, user_id: str, model_id: str) -> bool:
        """
        Erstellt Store für User/Model Kombination
        """
        await self.initialize()
        
        store_key = f"{user_id}:{model_id}"
        
        # Check cache first
        if store_key in self.stores:
            return self.stores[store_key]
        
        payload = {
            "user_id": user_id,
            "model_id": model_id
        }
        
        try:
            success = await self._make_request(
                "POST", 
                "/admin/create_store", 
                json=payload
            )
            
            self.stores[store_key] = success is not None
            return self.stores[store_key]
            
        except Exception as e:
            print(f"Error creating store for {user_id}:{model_id}: {e}")
            return False
    
    async def add_vectors(self, 
                         user_id: str, 
                         model_id: str,
                         vectors: Union[List[List[float]], mx.array, np.ndarray],
                         metadata: List[Dict[str, Any]],
                         ids: Optional[List[str]] = None,
                         namespace: Optional[str] = None) -> bool:
        """
        Fügt Vektoren zum Store hinzu
        """
        start_time = time.time()
        
        await self.initialize()
        
        # Ensure store exists
        await self.create_user_store(user_id, model_id)
        
        # Convert vectors to list format
        if isinstance(vectors, mx.array):
            vectors = vectors.tolist()
        elif isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()
        
        # Generate IDs if not provided
        if ids is None:
            ids = [self._generate_id(i, user_id, namespace) for i in range(len(vectors))]
        
        # Ensure metadata has same length
        if len(metadata) != len(vectors):
            raise ValueError(f"Metadata length ({len(metadata)}) must match vectors length ({len(vectors)})")
        
        # Add namespace and timestamp to metadata
        enriched_metadata = []
        current_time = datetime.now().isoformat()
        
        for i, meta in enumerate(metadata):
            enriched = meta.copy()
            enriched.update({
                "id": ids[i],
                "namespace": namespace,
                "timestamp": current_time,
                "user_id": user_id,
                "model_id": model_id
            })
            enriched_metadata.append(enriched)
        
        # Batch processing für große Mengen
        success = True
        for i in range(0, len(vectors), self.config.batch_size):
            batch_vectors = vectors[i:i + self.config.batch_size]
            batch_metadata = enriched_metadata[i:i + self.config.batch_size]
            
            payload = {
                "user_id": user_id,
                "model_id": model_id,
                "vectors": batch_vectors,
                "metadata": batch_metadata
            }
            
            try:
                result = await self._make_request(
                    "POST",
                    "/vectors/add",
                    json=payload
                )
                
                if result is None:
                    success = False
                    break
                    
            except Exception as e:
                print(f"Error adding vector batch {i}: {e}")
                success = False
                break
        
        # Update metrics
        add_time = time.time() - start_time
        self.add_count += len(vectors)
        self.total_add_time += add_time
        
        return success
    
    async def query(self, 
                   user_id: str, 
                   model_id: str,
                   query_vector: Union[List[float], mx.array, np.ndarray],
                   k: Optional[int] = None,
                   filters: Optional[Dict[str, Any]] = None,
                   namespace: Optional[str] = None,
                   include_vectors: bool = False) -> List[QueryResult]:
        """
        Führt Similarity Search durch
        """
        start_time = time.time()
        
        await self.initialize()
        
        # Convert query vector
        if isinstance(query_vector, (mx.array, np.ndarray)):
            query_vector = query_vector.tolist()
        
        k = k or self.config.default_k
        k = min(k, self.config.max_k)
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "query": query_vector,
            "k": k,
            "include_vectors": include_vectors
        }
        
        # Add filters if provided
        if filters:
            payload["filters"] = filters
        
        if namespace:
            if "filters" not in payload:
                payload["filters"] = {}
            payload["filters"]["namespace"] = namespace
        
        try:
            response = await self._make_request(
                "POST",
                "/vectors/query",
                json=payload
            )
            
            if response is None:
                return []
            
            # Parse results
            results = []
            for item in response.get("results", []):
                result = QueryResult(
                    id=item.get("id", ""),
                    score=item.get("score", 0.0),
                    metadata=item.get("metadata", {}),
                    vector=item.get("vector") if include_vectors else None
                )
                results.append(result)
            
            # Update metrics
            query_time = time.time() - start_time
            self.query_count += 1
            self.total_query_time += query_time
            
            return results
            
        except Exception as e:
            print(f"Error querying vectors: {e}")
            return []
    
    async def batch_query(self, 
                         user_id: str, 
                         model_id: str,
                         query_vectors: Union[List[List[float]], mx.array, np.ndarray],
                         k: Optional[int] = None,
                         filters: Optional[Dict[str, Any]] = None,
                         namespace: Optional[str] = None) -> List[List[QueryResult]]:
        """
        Batch Query Processing für multiple Queries
        """
        await self.initialize()
        
        # Convert query vectors
        if isinstance(query_vectors, (mx.array, np.ndarray)):
            query_vectors = query_vectors.tolist()
        
        k = k or self.config.default_k
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "queries": query_vectors,
            "k": k
        }
        
        if filters:
            payload["filters"] = filters
        
        if namespace:
            if "filters" not in payload:
                payload["filters"] = {}
            payload["filters"]["namespace"] = namespace
        
        try:
            response = await self._make_request(
                "POST",
                "/vectors/batch_query",
                json=payload
            )
            
            if response is None:
                return []
            
            # Parse batch results
            batch_results = []
            for batch_item in response.get("results", []):
                query_results = []
                for item in batch_item:
                    result = QueryResult(
                        id=item.get("id", ""),
                        score=item.get("score", 0.0),
                        metadata=item.get("metadata", {})
                    )
                    query_results.append(result)
                batch_results.append(query_results)
            
            return batch_results
            
        except Exception as e:
            print(f"Error in batch query: {e}")
            return []
    
    async def delete_vectors(self, 
                           user_id: str, 
                           model_id: str,
                           ids: List[str],
                           namespace: Optional[str] = None) -> bool:
        """
        Löscht Vektoren aus dem Store
        """
        await self.initialize()
        
        filters = {"id": {"$in": ids}}
        if namespace:
            filters["namespace"] = namespace
        
        payload = {
            "user_id": user_id,
            "model_id": model_id,
            "filters": filters
        }
        
        try:
            response = await self._make_request(
                "DELETE",
                "/vectors/delete",
                json=payload
            )
            
            return response is not None
            
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
    
    async def get_store_stats(self, user_id: str, model_id: str) -> Optional[VectorStoreStats]:
        """
        Holt Statistiken für einen Store
        """
        await self.initialize()
        
        try:
            response = await self._make_request(
                "GET",
                f"/vectors/count?user_id={user_id}&model_id={model_id}"
            )
            
            if response is None:
                return None
            
            return VectorStoreStats(
                total_vectors=response.get("count", 0),
                query_latency_ms=response.get("avg_query_time", 0.0) * 1000,
                add_latency_ms=response.get("avg_add_time", 0.0) * 1000,
                storage_size_mb=response.get("storage_size_mb", 0.0),
                last_updated=response.get("last_updated", "")
            )
            
        except Exception as e:
            print(f"Error getting store stats: {e}")
            return None
    
    async def export_store(self, user_id: str, model_id: str) -> Optional[bytes]:
        """
        Exportiert Store als ZIP
        """
        await self.initialize()
        
        try:
            async with self.session.get(
                f"{self.config.base_url}/admin/export_zip",
                params={"user_id": user_id, "model_id": model_id}
            ) as response:
                if response.status == 200:
                    return await response.read()
                return None
                
        except Exception as e:
            print(f"Error exporting store: {e}")
            return None
    
    async def import_store(self, user_id: str, model_id: str, zip_data: bytes) -> bool:
        """
        Importiert Store aus ZIP
        """
        await self.initialize()
        
        try:
            data = aiohttp.FormData()
            data.add_field('file', zip_data, filename='store.zip')
            data.add_field('user_id', user_id)
            data.add_field('model_id', model_id)
            
            async with self.session.post(
                f"{self.config.base_url}/admin/import_zip",
                data=data
            ) as response:
                return response.status == 200
                
        except Exception as e:
            print(f"Error importing store: {e}")
            return False
    
    async def _make_request(self, 
                          method: str, 
                          endpoint: str, 
                          json: Optional[Dict] = None,
                          params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Macht HTTP Request mit Retry Logic
        """
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.session.request(
                    method, 
                    url, 
                    json=json,
                    params=params
                ) as response:
                    if response.status == 200:
                        if response.content_type == 'application/json':
                            return await response.json()
                        else:
                            return {"success": True}
                    elif response.status == 404:
                        print(f"Endpoint not found: {endpoint}")
                        return None
                    elif response.status == 401:
                        print(f"Authentication failed for {endpoint}")
                        return None
                    else:
                        print(f"HTTP {response.status} for {endpoint}")
                        if attempt < self.config.retry_attempts - 1:
                            await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                            continue
                        return None
                        
            except asyncio.TimeoutError:
                print(f"Timeout for {endpoint} (attempt {attempt + 1})")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                return None
            except Exception as e:
                print(f"Request error for {endpoint}: {e}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                return None
        
        return None
    
    async def _health_check(self) -> bool:
        """
        Prüft ob mlx-vector-db erreichbar ist
        """
        try:
            response = await self._make_request("GET", "/monitoring/health")
            if response:
                print(f"✅ mlx-vector-db connection established")
                return True
            else:
                print(f"❌ mlx-vector-db health check failed")
                return False
                
        except Exception as e:
            print(f"❌ Cannot connect to mlx-vector-db: {e}")
            return False
    
    def _generate_id(self, index: int, user_id: str, namespace: Optional[str] = None) -> str:
        """
        Generiert eindeutige ID für Vektor
        """
        timestamp = int(time.time() * 1000)
        content = f"{user_id}:{namespace}:{index}:{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Liefert Performance-Statistiken
        """
        avg_query_time = self.total_query_time / self.query_count if self.query_count > 0 else 0
        avg_add_time = self.total_add_time / self.add_count if self.add_count > 0 else 0
        
        return {
            "total_queries": self.query_count,
            "total_vectors_added": self.add_count,
            "average_query_time_ms": avg_query_time * 1000,
            "average_add_time_ms": avg_add_time * 1000,
            "queries_per_second": self.query_count / self.total_query_time if self.total_query_time > 0 else 0,
            "vectors_per_second": self.add_count / self.total_add_time if self.total_add_time > 0 else 0,
            "active_stores": len(self.stores)
        }
    
    async def benchmark(self, user_id: str = "benchmark_user", model_id: str = "test_model") -> Dict[str, float]:
        """
        Performance Benchmark für Vector Store
        """
        print("Running Vector Store Benchmark...")
        
        # Test data
        test_vectors = np.random.rand(100, 384).astype(np.float32).tolist()
        test_metadata = [{"doc_id": f"doc_{i}", "content": f"test document {i}"} for i in range(100)]
        
        # Create store
        await self.create_user_store(user_id, model_id)
        
        # Benchmark add operations
        start_time = time.time()
        success = await self.add_vectors(user_id, model_id, test_vectors, test_metadata)
        add_time = time.time() - start_time
        
        if not success:
            return {"error": "Failed to add vectors"}
        
        # Benchmark query operations
        query_times = []
        for i in range(10):
            query_vector = test_vectors[i]
            
            start_time = time.time()
            results = await self.query(user_id, model_id, query_vector, k=5)
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        avg_query_time = sum(query_times) / len(query_times)
        
        # Benchmark batch query
        batch_queries = test_vectors[:10]
        start_time = time.time()
        batch_results = await self.batch_query(user_id, model_id, batch_queries, k=5)
        batch_time = time.time() - start_time
        
        return {
            "vectors_added": len(test_vectors),
            "add_time_seconds": add_time,
            "vectors_per_second": len(test_vectors) / add_time,
            "average_query_time_ms": avg_query_time * 1000,
            "queries_per_second": 1 / avg_query_time,
            "batch_query_time_seconds": batch_time,
            "batch_queries_per_second": len(batch_queries) / batch_time
        }

# Usage Examples
async def example_usage():
    """Beispiele für Vector Store Usage"""
    
    # Initialize with config
    config = VectorStoreConfig(
        base_url="http://localhost:8000",
        api_key="your-api-key",
        batch_size=50
    )
    
    store = MLXVectorStore(config)
    
    try:
        # Create user store
        success = await store.create_user_store("user_123", "gte-small")
        print(f"Store created: {success}")
        
        # Add vectors
        vectors = np.random.rand(10, 384).tolist()
        metadata = [{"text": f"Document {i}", "category": "test"} for i in range(10)]
        
        success = await store.add_vectors("user_123", "gte-small", vectors, metadata)
        print(f"Vectors added: {success}")
        
        # Query vectors
        query_vector = vectors[0]
        results = await store.query("user_123", "gte-small", query_vector, k=5)
        print(f"Query results: {len(results)}")
        
        for result in results[:3]:
            print(f"  ID: {result.id}, Score: {result.score:.3f}")
        
        # Performance stats
        stats = store.get_performance_stats()
        print(f"Performance: {stats}")
        
        # Benchmark
        benchmark_results = await store.benchmark()
        print(f"Benchmark: {benchmark_results}")
        
    finally:
        await store.close()

if __name__ == "__main__":
    asyncio.run(example_usage())