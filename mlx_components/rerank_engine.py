"""
MLX Re-ranking Engine
Optimiert Search Results zwischen Vector Retrieval und LLM Generation
Verbessert Relevanz und Qualität der RAG Responses
"""

import asyncio
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from collections import Counter

@dataclass
class ReRankConfig:
    """Konfiguration für Re-ranking Engine"""
    rerank_model_path: Optional[str] = None  # Falls MLX Re-rank Modell verfügbar
    max_candidates: int = 50
    top_k: int = 5
    score_threshold: float = 0.1
    diversity_factor: float = 0.3
    freshness_weight: float = 0.1
    length_penalty: float = 0.05
    semantic_boost: float = 0.2
    enable_diversity_rerank: bool = True
    enable_semantic_clustering: bool = True

@dataclass
class RerankCandidate:
    """Kandidat für Re-ranking"""
    id: str
    content: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: Optional[float] = None
    final_score: Optional[float] = None
    features: Optional[Dict[str, float]] = None

@dataclass
class RerankResult:
    """Ergebnis des Re-ranking Prozesses"""
    candidates: List[RerankCandidate]
    total_candidates: int
    processing_time: float
    algorithm_used: str
    score_distribution: Dict[str, float]

class MLXRerankEngine:
    """
    Intelligente Re-ranking Engine für RAG Optimization
    
    Features:
    - Multiple Re-ranking Algorithms
    - Semantic Diversity Optimization
    - Query-Document Relevance Scoring
    - Freshness and Length Balancing
    - Content Clustering für Diversity
    - Performance Monitoring
    - Hybrid Scoring Models
    """
    
    def __init__(self, config: ReRankConfig = None):
        self.config = config or ReRankConfig()
        self.rerank_model = None
        self.tokenizer = None
        
        # Performance Metrics
        self.total_reranks = 0
        self.total_processing_time = 0.0
        self.algorithm_usage = Counter()
        
        # Feature extractors
        self.feature_extractors = {
            "query_overlap": self._extract_query_overlap,
            "content_length": self._extract_content_length,
            "freshness": self._extract_freshness,
            "semantic_density": self._extract_semantic_density,
            "metadata_relevance": self._extract_metadata_relevance
        }
        
        # Available algorithms
        self.algorithms = {
            "hybrid_scoring": self._hybrid_scoring_rerank,
            "semantic_diversity": self._semantic_diversity_rerank,
            "query_focused": self._query_focused_rerank,
            "balanced": self._balanced_rerank
        }
    
    async def initialize(self) -> None:
        """
        Initialisiert Re-rank Modell falls verfügbar
        """
        if self.config.rerank_model_path and self.rerank_model is None:
            try:
                print(f"Loading rerank model: {self.config.rerank_model_path}")
                # Placeholder für zukünftige MLX Re-rank Modelle
                # self.rerank_model, self.tokenizer = load(self.config.rerank_model_path)
                print("✅ Rerank model loaded (placeholder)")
            except Exception as e:
                print(f"⚠️ Rerank model not available, using algorithmic ranking: {e}")
    
    async def rerank(self, 
                    query: str,
                    candidates: List[Dict[str, Any]],
                    top_k: Optional[int] = None,
                    algorithm: str = "balanced") -> RerankResult:
        """
        Hauptfunktion: Re-rankt Kandidaten basierend auf Query
        """
        start_time = time.time()
        
        await self.initialize()
        
        top_k = top_k or self.config.top_k
        
        # Convert to RerankCandidate objects
        rerank_candidates = []
        for i, candidate in enumerate(candidates):
            rerank_candidate = RerankCandidate(
                id=candidate.get("id", f"candidate_{i}"),
                content=candidate.get("content", candidate.get("text", "")),
                metadata=candidate.get("metadata", {}),
                original_score=candidate.get("score", 0.0)
            )
            rerank_candidates.append(rerank_candidate)
        
        # Limit candidates if too many
        if len(rerank_candidates) > self.config.max_candidates:
            # Keep top candidates by original score
            rerank_candidates.sort(key=lambda x: x.original_score, reverse=True)
            rerank_candidates = rerank_candidates[:self.config.max_candidates]
        
        # Extract features for all candidates
        for candidate in rerank_candidates:
            candidate.features = await self._extract_features(query, candidate)
        
        # Apply selected algorithm
        if algorithm in self.algorithms:
            ranked_candidates = await self.algorithms[algorithm](query, rerank_candidates)
        else:
            print(f"Unknown algorithm {algorithm}, using balanced")
            ranked_candidates = await self._balanced_rerank(query, rerank_candidates)
        
        # Filter by score threshold and limit
        filtered_candidates = [
            c for c in ranked_candidates 
            if c.final_score >= self.config.score_threshold
        ][:top_k]
        
        processing_time = time.time() - start_time
        
        # Update metrics
        self.total_reranks += 1
        self.total_processing_time += processing_time
        self.algorithm_usage[algorithm] += 1
        
        # Calculate score distribution
        scores = [c.final_score for c in filtered_candidates if c.final_score is not None]
        score_distribution = {
            "mean": np.mean(scores) if scores else 0.0,
            "std": np.std(scores) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0
        }
        
        return RerankResult(
            candidates=filtered_candidates,
            total_candidates=len(candidates),
            processing_time=processing_time,
            algorithm_used=algorithm,
            score_distribution=score_distribution
        )
    
    async def _hybrid_scoring_rerank(self, 
                                   query: str, 
                                   candidates: List[RerankCandidate]) -> List[RerankCandidate]:
        """
        Hybrid Scoring kombiniert multiple Faktoren
        """
        for candidate in candidates:
            features = candidate.features
            
            # Weighted combination of features
            hybrid_score = (
                features["query_overlap"] * 0.4 +
                features["semantic_density"] * 0.25 +
                candidate.original_score * 0.2 +
                features["freshness"] * self.config.freshness_weight +
                features["metadata_relevance"] * 0.1 -
                features["length_penalty"] * self.config.length_penalty
            )
            
            candidate.rerank_score = hybrid_score
            candidate.final_score = hybrid_score
        
        # Sort by final score
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates
    
    async def _semantic_diversity_rerank(self, 
                                       query: str, 
                                       candidates: List[RerankCandidate]) -> List[RerankCandidate]:
        """
        Semantic Diversity Reranking für vielfältige Results
        """
        if not self.config.enable_diversity_rerank:
            return await self._hybrid_scoring_rerank(query, candidates)
        
        # First pass: Score by relevance
        relevance_scored = await self._hybrid_scoring_rerank(query, candidates)
        
        # Second pass: Apply diversity penalty
        final_candidates = []
        selected_content = []
        
        for candidate in relevance_scored:
            # Calculate diversity penalty
            diversity_penalty = 0.0
            
            for selected in selected_content:
                similarity = self._calculate_content_similarity(
                    candidate.content, 
                    selected
                )
                diversity_penalty += similarity * self.config.diversity_factor
            
            # Apply diversity penalty
            diversity_score = candidate.rerank_score - diversity_penalty
            candidate.final_score = diversity_score
            
            final_candidates.append(candidate)
            selected_content.append(candidate.content)
        
        # Sort by diversity-adjusted score
        final_candidates.sort(key=lambda x: x.final_score, reverse=True)
        return final_candidates
    
    async def _query_focused_rerank(self, 
                                  query: str, 
                                  candidates: List[RerankCandidate]) -> List[RerankCandidate]:
        """
        Query-focused Reranking priorisiert Query-Ähnlichkeit
        """
        for candidate in candidates:
            features = candidate.features
            
            # Strong focus on query relevance
            query_score = (
                features["query_overlap"] * 0.6 +
                features["semantic_density"] * 0.3 +
                candidate.original_score * 0.1
            )
            
            candidate.rerank_score = query_score
            candidate.final_score = query_score
        
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates
    
    async def _balanced_rerank(self, 
                             query: str, 
                             candidates: List[RerankCandidate]) -> List[RerankCandidate]:
        """
        Balanced Reranking kombiniert Relevanz und Diversity
        """
        # Apply hybrid scoring first
        hybrid_candidates = await self._hybrid_scoring_rerank(query, candidates)
        
        # Then apply light diversity adjustment
        if self.config.enable_diversity_rerank:
            # Use lower diversity factor for balanced approach
            original_diversity_factor = self.config.diversity_factor
            self.config.diversity_factor *= 0.5  # Reduce diversity impact
            
            diversity_candidates = await self._semantic_diversity_rerank(query, hybrid_candidates)
            
            # Restore original setting
            self.config.diversity_factor = original_diversity_factor
            
            return diversity_candidates
        
        return hybrid_candidates
    
    async def _extract_features(self, 
                              query: str, 
                              candidate: RerankCandidate) -> Dict[str, float]:
        """
        Extrahiert Features für Scoring
        """
        features = {}
        
        for feature_name, extractor in self.feature_extractors.items():
            try:
                features[feature_name] = await extractor(query, candidate)
            except Exception as e:
                print(f"Error extracting feature {feature_name}: {e}")
                features[feature_name] = 0.0
        
        return features
    
    async def _extract_query_overlap(self, 
                                   query: str, 
                                   candidate: RerankCandidate) -> float:
        """
        Berechnet Query-Content Overlap
        """
        query_words = set(query.lower().split())
        content_words = set(candidate.content.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)
    
    async def _extract_content_length(self, 
                                    query: str, 
                                    candidate: RerankCandidate) -> float:
        """
        Normalisiert Content Length (längere Texte haben penalties)
        """
        content_length = len(candidate.content)
        
        # Optimal length around 500-1000 characters
        optimal_length = 750
        
        if content_length <= optimal_length:
            return 1.0
        else:
            # Penalty for overly long content
            penalty = min(1.0, (content_length - optimal_length) / optimal_length)
            return max(0.0, 1.0 - penalty)
    
    async def _extract_freshness(self, 
                               query: str, 
                               candidate: RerankCandidate) -> float:
        """
        Bewertet Content Freshness basierend auf Timestamps
        """
        timestamp_str = candidate.metadata.get("timestamp")
        if not timestamp_str:
            return 0.5  # Neutral score for unknown timestamps
        
        try:
            content_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            current_time = datetime.now()
            
            # Calculate age in days
            age_days = (current_time - content_time).days
            
            # Fresher content gets higher scores
            if age_days <= 1:
                return 1.0
            elif age_days <= 7:
                return 0.8
            elif age_days <= 30:
                return 0.6
            elif age_days <= 90:
                return 0.4
            else:
                return 0.2
                
        except Exception:
            return 0.5
    
    async def _extract_semantic_density(self, 
                                      query: str, 
                                      candidate: RerankCandidate) -> float:
        """
        Schätzt semantische Dichte des Contents
        """
        content = candidate.content
        
        # Simple heuristics for semantic density
        sentences = content.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Optimal sentence length around 10-20 words
        if 8 <= avg_sentence_length <= 25:
            sentence_score = 1.0
        else:
            sentence_score = max(0.3, 1.0 - abs(avg_sentence_length - 15) / 20)
        
        # Check for technical terms or specific vocabulary
        technical_terms = len([w for w in content.split() if len(w) > 6])
        technical_ratio = technical_terms / len(content.split()) if content.split() else 0
        
        # Balance between technical depth and readability
        technical_score = min(1.0, technical_ratio * 3)
        
        return (sentence_score + technical_score) / 2
    
    async def _extract_metadata_relevance(self, 
                                        query: str, 
                                        candidate: RerankCandidate) -> float:
        """
        Bewertet Metadata-Relevanz zur Query
        """
        metadata = candidate.metadata
        query_lower = query.lower()
        
        relevance_score = 0.0
        
        # Check title relevance
        title = metadata.get("title", "").lower()
        if title and any(word in title for word in query_lower.split()):
            relevance_score += 0.4
        
        # Check category/topic relevance
        category = metadata.get("category", "").lower()
        if category and any(word in category for word in query_lower.split()):
            relevance_score += 0.3
        
        # Check tags relevance
        tags = metadata.get("tags", [])
        if tags:
            tag_matches = sum(1 for tag in tags if any(word in tag.lower() for word in query_lower.split()))
            relevance_score += min(0.3, tag_matches * 0.1)
        
        return min(1.0, relevance_score)
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Berechnet Ähnlichkeit zwischen zwei Texten
        """
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    async def rerank_with_llm_scoring(self, 
                                    query: str,
                                    candidates: List[Dict[str, Any]],
                                    llm_handler,
                                    top_k: Optional[int] = None) -> RerankResult:
        """
        Advanced Re-ranking mit LLM-basiertem Scoring
        """
        top_k = top_k or self.config.top_k
        
        # First pass: Standard reranking
        initial_rerank = await self.rerank(query, candidates, top_k * 2, "balanced")
        
        # Second pass: LLM scoring for top candidates
        llm_scored_candidates = []
        
        for candidate in initial_rerank.candidates[:top_k * 2]:
            # Create LLM prompt for relevance scoring
            scoring_prompt = f"""
            Rate the relevance of the following document to the query on a scale of 0.0 to 1.0.
            
            Query: {query}
            
            Document: {candidate.content[:500]}...
            
            Consider: semantic relevance, factual accuracy, and completeness.
            Respond with only a number between 0.0 and 1.0.
            """
            
            try:
                from mlx_components.llm_handler import LLMRequest
                
                llm_request = LLMRequest(
                    prompt=scoring_prompt,
                    user_id="rerank_system",
                    max_tokens=10,
                    temperature=0.0
                )
                
                llm_response = await llm_handler.generate_single(llm_request)
                
                # Parse LLM score
                llm_score_str = llm_response.response.strip()
                try:
                    llm_score = float(llm_score_str)
                    llm_score = max(0.0, min(1.0, llm_score))  # Clamp to [0,1]
                except ValueError:
                    llm_score = candidate.final_score  # Fallback to original score
                
                # Combine original score with LLM score
                combined_score = (candidate.final_score * 0.6) + (llm_score * 0.4)
                candidate.final_score = combined_score
                candidate.metadata["llm_score"] = llm_score
                
                llm_scored_candidates.append(candidate)
                
            except Exception as e:
                print(f"Error in LLM scoring: {e}")
                llm_scored_candidates.append(candidate)
        
        # Sort by combined score
        llm_scored_candidates.sort(key=lambda x: x.final_score, reverse=True)
        
        return RerankResult(
            candidates=llm_scored_candidates[:top_k],
            total_candidates=len(candidates),
            processing_time=initial_rerank.processing_time,
            algorithm_used="llm_enhanced",
            score_distribution=initial_rerank.score_distribution
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Liefert Performance-Statistiken
        """
        avg_processing_time = self.total_processing_time / self.total_reranks if self.total_reranks > 0 else 0
        
        return {
            "total_reranks": self.total_reranks,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "reranks_per_second": self.total_reranks / self.total_processing_time if self.total_processing_time > 0 else 0,
            "algorithm_usage": dict(self.algorithm_usage),
            "available_algorithms": list(self.algorithms.keys()),
            "feature_extractors": list(self.feature_extractors.keys())
        }
    
    async def benchmark(self, sample_query: str = "machine learning applications") -> Dict[str, float]:
        """
        Performance Benchmark für Re-ranking Engine
        """
        print("Running Re-ranking Engine Benchmark...")
        
        # Create test candidates
        test_candidates = []
        for i in range(20):
            candidate = {
                "id": f"doc_{i}",
                "content": f"This is test document {i} about machine learning and artificial intelligence. " * 10,
                "metadata": {
                    "title": f"Document {i}",
                    "timestamp": datetime.now().isoformat(),
                    "category": "technology" if i % 2 == 0 else "research"
                },
                "score": 0.8 - (i * 0.02)  # Decreasing scores
            }
            test_candidates.append(candidate)
        
        # Benchmark different algorithms
        results = {}
        
        for algorithm in self.algorithms.keys():
            start_time = time.time()
            
            rerank_result = await self.rerank(
                query=sample_query,
                candidates=test_candidates,
                top_k=10,
                algorithm=algorithm
            )
            
            algorithm_time = time.time() - start_time
            results[f"{algorithm}_time"] = algorithm_time
            results[f"{algorithm}_candidates"] = len(rerank_result.candidates)
        
        return results

# Usage Examples
async def example_usage():
    """Beispiele für Re-ranking Engine Usage"""
    
    # Initialize with config
    config = ReRankConfig(
        top_k=5,
        diversity_factor=0.3,
        enable_diversity_rerank=True
    )
    
    rerank_engine = MLXRerankEngine(config)
    
    # Test candidates (simulate vector search results)
    candidates = [
        {
            "id": "doc_1",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            "metadata": {"title": "ML Basics", "timestamp": "2024-01-01T00:00:00"},
            "score": 0.85
        },
        {
            "id": "doc_2", 
            "content": "Neural networks are computing systems inspired by biological neural networks.",
            "metadata": {"title": "Neural Networks", "timestamp": "2024-01-02T00:00:00"},
            "score": 0.82
        },
        {
            "id": "doc_3",
            "content": "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "metadata": {"title": "Deep Learning", "timestamp": "2024-01-03T00:00:00"},
            "score": 0.80
        }
    ]
    
    # Standard reranking
    query = "What is machine learning?"
    
    rerank_result = await rerank_engine.rerank(
        query=query,
        candidates=candidates,
        top_k=5,
        algorithm="balanced"
    )
    
    print(f"Reranked {rerank_result.total_candidates} candidates")
    print(f"Processing time: {rerank_result.processing_time:.3f}s")
    print(f"Algorithm used: {rerank_result.algorithm_used}")
    
    for i, candidate in enumerate(rerank_result.candidates):
        print(f"  {i+1}. {candidate.id}: {candidate.final_score:.3f} (was {candidate.original_score:.3f})")
    
    # Performance stats
    stats = rerank_engine.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Benchmark
    benchmark_results = await rerank_engine.benchmark()
    print(f"Benchmark results: {benchmark_results}")

if __name__ == "__main__":
    asyncio.run(example_usage())