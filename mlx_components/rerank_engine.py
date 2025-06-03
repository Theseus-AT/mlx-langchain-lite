"""
MLX Rerank Engine - LLM-basiertes Batch-Reranking
Intelligente Neuordnung von Suchergebnissen mit MLX Parallels Integration
"""

import asyncio
import time
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import math

# MLX Parallels Integration
try:
    from mlx_parallels.core.batch_processor import BatchProcessor, BatchResult
    from mlx_parallels.core.config import get_fast_inference_config
    BATCH_PROCESSING_AVAILABLE = True
except ImportError:
    BATCH_PROCESSING_AVAILABLE = False

logger = logging.getLogger(__name__)


class RerankMethod(Enum):
    """Reranking-Methoden"""
    SIMILARITY_SCORE = "similarity_score"
    BM25 = "bm25"
    TFIDF = "tfidf"
    LLM_SCORING = "llm_scoring"
    LLM_PAIRWISE = "llm_pairwise"
    HYBRID = "hybrid"


@dataclass
class RerankConfig:
    """Konfiguration für Reranking Engine"""
    
    # Haupt-Reranking Methode
    primary_method: RerankMethod = RerankMethod.HYBRID
    fallback_method: RerankMethod = RerankMethod.SIMILARITY_SCORE
    
    # LLM-Reranking Settings
    llm_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    enable_batch_llm: bool = True
    max_llm_batch_size: int = 16
    llm_temperature: float = 0.1  # Niedrig für konsistente Scores
    
    # Scoring Settings
    max_documents_for_llm: int = 20  # LLM nur für Top-K
    score_threshold: float = 0.1
    diversity_factor: float = 0.1  # Für Diversität in Ergebnissen
    
    # Performance Settings
    enable_caching: bool = True
    cache_size: int = 1000
    timeout_seconds: float = 30.0
    
    # BM25/TF-IDF Settings
    k1: float = 1.5  # BM25 parameter
    b: float = 0.75  # BM25 parameter
    min_doc_freq: int = 1


@dataclass
class RerankResult:
    """Reranking-Ergebnis"""
    documents: List[str]
    scores: List[float]
    original_indices: List[int]
    method_used: RerankMethod
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchRerankResult:
    """Batch-Reranking-Ergebnis"""
    results: List[RerankResult]
    total_queries: int
    processing_time: float
    avg_docs_per_query: float
    successful_queries: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLXRerankEngine:
    """
    Enhanced Rerank Engine mit LLM-basiertem Batch-Reranking
    
    Features:
    - Multiple Reranking-Algorithmen (BM25, TF-IDF, LLM-Scoring)
    - Batch-LLM-Reranking für bessere Performance
    - Hybrid-Ansatz kombiniert verschiedene Methoden
    - Intelligent Caching für wiederholte Queries
    - Diversitäts-Optimierung
    """
    
    def __init__(self, config: RerankConfig):
        self.config = config
        self.llm_handler = None
        self.cache = {} if config.enable_caching else None
        
        # Performance Tracking
        self.stats = {
            'total_reranks': 0,
            'batch_reranks': 0,
            'llm_reranks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time': 0.0,
            'method_usage': {method.value: 0 for method in RerankMethod}
        }
        
        # Pre-computed word frequencies für BM25/TF-IDF
        self._doc_frequencies = {}
        self._avg_doc_length = 0
        
        logger.info(f"RerankEngine initialisiert - Primäre Methode: {config.primary_method.value}")
    
    async def initialize(self, llm_handler=None):
        """Initialisiert Rerank Engine"""
        try:
            # LLM Handler für LLM-basiertes Reranking
            if llm_handler:
                self.llm_handler = llm_handler
                logger.info("✅ Externe LLM Handler verbunden")
            elif self.config.enable_batch_llm and BATCH_PROCESSING_AVAILABLE:
                # Eigener LLM Handler für Reranking
                mlx_config = get_fast_inference_config(self.config.llm_model)
                mlx_config.generation.temperature = self.config.llm_temperature
                mlx_config.generation.max_tokens = 50  # Kurze Scores
                mlx_config.batch.max_batch_size = self.config.max_llm_batch_size
                
                self.batch_processor = BatchProcessor(mlx_config)
                success = self.batch_processor.load_model()
                
                if success:
                    logger.info("✅ Integrierte LLM Handler für Reranking initialisiert")
                else:
                    logger.warning("⚠️  LLM Handler fehlgeschlagen - verwende Fallback-Methoden")
                    self.batch_processor = None
            else:
                logger.info("📝 Reranking ohne LLM-Unterstützung")
                self.batch_processor = None
            
            return True
            
        except Exception as e:
            logger.error(f"Rerank Engine Initialisierung fehlgeschlagen: {e}")
            return False
    
    # Haupt-Reranking Methoden
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None,
        method: Optional[RerankMethod] = None
    ) -> List[str]:
        """
        Einzelnes Reranking - Legacy-kompatibel
        """
        result = await self.rerank_with_scores(query, documents, scores, method)
        return result.documents
    
    async def rerank_with_scores(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]] = None,
        method: Optional[RerankMethod] = None
    ) -> RerankResult:
        """
        Reranking mit detaillierten Scores
        """
        if not documents:
            return RerankResult(
                documents=[],
                scores=[],
                original_indices=[],
                method_used=method or self.config.primary_method,
                processing_time=0.0
            )
        
        method = method or self.config.primary_method
        start_time = time.time()
        
        # Cache-Check
        cache_key = None
        if self.cache is not None:
            cache_key = self._generate_cache_key(query, documents, method)
            if cache_key in self.cache:
                self.stats['cache_hits'] += 1
                cached_result = self.cache[cache_key]
                cached_result.metadata['from_cache'] = True
                return cached_result
            self.stats['cache_misses'] += 1
        
        try:
            # Reranking durchführen
            result = await self._execute_reranking(query, documents, scores, method)
            
            # Cache speichern
            if self.cache is not None and cache_key:
                if len(self.cache) < self.config.cache_size:
                    self.cache[cache_key] = result
                elif len(self.cache) >= self.config.cache_size:
                    # LRU: Entferne ältesten Eintrag
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    self.cache[cache_key] = result
            
            # Stats aktualisieren
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            self._update_stats(1, processing_time, method)
            
            return result
            
        except Exception as e:
            logger.error(f"Reranking fehlgeschlagen: {e}")
            # Fallback: Original-Reihenfolge
            return RerankResult(
                documents=documents,
                scores=scores or [1.0] * len(documents),
                original_indices=list(range(len(documents))),
                method_used=RerankMethod.SIMILARITY_SCORE,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    async def batch_rerank(
        self,
        queries: List[str],
        document_lists: List[List[str]],
        score_lists: Optional[List[List[float]]] = None,
        method: Optional[RerankMethod] = None
    ) -> BatchRerankResult:
        """
        Batch-Reranking für optimale Performance
        """
        if len(queries) != len(document_lists):
            raise ValueError("Anzahl Queries muss Anzahl Document-Listen entsprechen")
        
        method = method or self.config.primary_method
        start_time = time.time()
        
        try:
            # Batch-LLM-Reranking falls möglich
            if method in [RerankMethod.LLM_SCORING, RerankMethod.LLM_PAIRWISE, RerankMethod.HYBRID]:
                if self._can_use_batch_llm():
                    results = await self._batch_llm_rerank(
                        queries, document_lists, score_lists, method
                    )
                else:
                    # Fallback: Sequentiell
                    results = await self._sequential_rerank(
                        queries, document_lists, score_lists, method
                    )
            else:
                # Non-LLM Methoden parallel verarbeiten
                results = await self._parallel_non_llm_rerank(
                    queries, document_lists, score_lists, method
                )
            
            processing_time = time.time() - start_time
            successful_queries = len([r for r in results if 'error' not in r.metadata])
            
            # Stats aktualisieren
            self.stats['batch_reranks'] += 1
            self._update_stats(len(queries), processing_time, method)
            
            # Durchschnittliche Dokumente pro Query
            total_docs = sum(len(docs) for docs in document_lists)
            avg_docs = total_docs / len(document_lists) if document_lists else 0
            
            logger.info(f"Batch-Reranking abgeschlossen: {len(queries)} Queries, "
                       f"{successful_queries} erfolgreich, {processing_time:.2f}s")
            
            return BatchRerankResult(
                results=results,
                total_queries=len(queries),
                processing_time=processing_time,
                avg_docs_per_query=avg_docs,
                successful_queries=successful_queries,
                metadata={
                    'method': method.value,
                    'batch_llm_used': method in [RerankMethod.LLM_SCORING, RerankMethod.LLM_PAIRWISE] and self._can_use_batch_llm()
                }
            )
            
        except Exception as e:
            logger.error(f"Batch-Reranking fehlgeschlagen: {e}")
            # Fallback: Leere Ergebnisse
            empty_results = []
            for i, (query, docs) in enumerate(zip(queries, document_lists)):
                empty_results.append(RerankResult(
                    documents=docs,
                    scores=[1.0] * len(docs),
                    original_indices=list(range(len(docs))),
                    method_used=RerankMethod.SIMILARITY_SCORE,
                    processing_time=0.0,
                    metadata={'error': str(e)}
                ))
            
            return BatchRerankResult(
                results=empty_results,
                total_queries=len(queries),
                processing_time=time.time() - start_time,
                avg_docs_per_query=0,
                successful_queries=0,
                metadata={'error': str(e)}
            )
    
    # Spezifische Reranking-Algorithmen
    
    async def _execute_reranking(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]],
        method: RerankMethod
    ) -> RerankResult:
        """Führt spezifische Reranking-Methode aus"""
        
        original_indices = list(range(len(documents)))
        
        if method == RerankMethod.SIMILARITY_SCORE:
            return await self._similarity_rerank(query, documents, scores)
        elif method == RerankMethod.BM25:
            return await self._bm25_rerank(query, documents, scores)
        elif method == RerankMethod.TFIDF:
            return await self._tfidf_rerank(query, documents, scores)
        elif method == RerankMethod.LLM_SCORING:
            return await self._llm_scoring_rerank(query, documents, scores)
        elif method == RerankMethod.LLM_PAIRWISE:
            return await self._llm_pairwise_rerank(query, documents, scores)
        elif method == RerankMethod.HYBRID:
            return await self._hybrid_rerank(query, documents, scores)
        else:
            # Fallback: Similarity-basiert
            return await self._similarity_rerank(query, documents, scores)
    
    async def _similarity_rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]]
    ) -> RerankResult:
        """Einfaches Score-basiertes Reranking"""
        
        if scores is None:
            scores = [1.0] * len(documents)
        
        # Sortiere nach Scores (absteigend)
        sorted_items = sorted(
            zip(documents, scores, range(len(documents))),
            key=lambda x: x[1],
            reverse=True
        )
        
        reranked_docs, reranked_scores, original_indices = zip(*sorted_items)
        
        return RerankResult(
            documents=list(reranked_docs),
            scores=list(reranked_scores),
            original_indices=list(original_indices),
            method_used=RerankMethod.SIMILARITY_SCORE,
            processing_time=0.0
        )
    
    async def _bm25_rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]]
    ) -> RerankResult:
        """BM25-basiertes Reranking"""
        
        try:
            # Query-Terms extrahieren
            query_terms = self._tokenize(query.lower())
            
            # BM25-Scores berechnen
            bm25_scores = []
            doc_lengths = [len(self._tokenize(doc)) for doc in documents]
            avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1
            
            for i, document in enumerate(documents):
                doc_terms = self._tokenize(document.lower())
                doc_length = doc_lengths[i]
                
                score = 0.0
                for term in query_terms:
                    # Term frequency in document
                    tf = doc_terms.count(term)
                    if tf == 0:
                        continue
                    
                    # Document frequency (vereinfacht)
                    df = sum(1 for doc in documents if term in self._tokenize(doc.lower()))
                    idf = math.log((len(documents) - df + 0.5) / (df + 0.5))
                    
                    # BM25 Score
                    numerator = tf * (self.config.k1 + 1)
                    denominator = tf + self.config.k1 * (1 - self.config.b + self.config.b * (doc_length / avg_doc_length))
                    
                    score += idf * (numerator / denominator)
                
                bm25_scores.append(score)
            
            # Kombiniere mit Original-Scores falls vorhanden
            if scores:
                # Normalisiere beide Score-Sets
                max_bm25 = max(bm25_scores) if bm25_scores else 1
                max_orig = max(scores) if scores else 1
                
                combined_scores = [
                    0.7 * (bm25 / max_bm25) + 0.3 * (orig / max_orig)
                    for bm25, orig in zip(bm25_scores, scores)
                ]
            else:
                combined_scores = bm25_scores
            
            # Sortieren
            sorted_items = sorted(
                zip(documents, combined_scores, range(len(documents))),
                key=lambda x: x[1],
                reverse=True
            )
            
            reranked_docs, reranked_scores, original_indices = zip(*sorted_items)
            
            return RerankResult(
                documents=list(reranked_docs),
                scores=list(reranked_scores),
                original_indices=list(original_indices),
                method_used=RerankMethod.BM25,
                processing_time=0.0,
                metadata={'avg_doc_length': avg_doc_length, 'query_terms': len(query_terms)}
            )
            
        except Exception as e:
            logger.warning(f"BM25 Reranking fehlgeschlagen: {e}")
            return await self._similarity_rerank(query, documents, scores)
    
    async def _tfidf_rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]]
    ) -> RerankResult:
        """TF-IDF-basiertes Reranking"""
        
        try:
            # Query-Terms
            query_terms = self._tokenize(query.lower())
            
            # TF-IDF Scores berechnen
            tfidf_scores = []
            
            for document in documents:
                doc_terms = self._tokenize(document.lower())
                doc_length = len(doc_terms)
                
                score = 0.0
                for term in query_terms:
                    # Term frequency
                    tf = doc_terms.count(term) / doc_length if doc_length > 0 else 0
                    
                    # Inverse document frequency
                    df = sum(1 for doc in documents if term in self._tokenize(doc.lower()))
                    idf = math.log(len(documents) / (df + 1))
                    
                    score += tf * idf
                
                tfidf_scores.append(score)
            
            # Kombiniere mit Original-Scores
            if scores:
                max_tfidf = max(tfidf_scores) if tfidf_scores else 1
                max_orig = max(scores) if scores else 1
                
                combined_scores = [
                    0.6 * (tfidf / max_tfidf) + 0.4 * (orig / max_orig)
                    for tfidf, orig in zip(tfidf_scores, scores)
                ]
            else:
                combined_scores = tfidf_scores
            
            # Sortieren
            sorted_items = sorted(
                zip(documents, combined_scores, range(len(documents))),
                key=lambda x: x[1],
                reverse=True
            )
            
            reranked_docs, reranked_scores, original_indices = zip(*sorted_items)
            
            return RerankResult(
                documents=list(reranked_docs),
                scores=list(reranked_scores),
                original_indices=list(original_indices),
                method_used=RerankMethod.TFIDF,
                processing_time=0.0,
                metadata={'query_terms': len(query_terms)}
            )
            
        except Exception as e:
            logger.warning(f"TF-IDF Reranking fehlgeschlagen: {e}")
            return await self._similarity_rerank(query, documents, scores)
    
    async def _llm_scoring_rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]]
    ) -> RerankResult:
        """LLM-basiertes Scoring Reranking"""
        
        if not self._can_use_llm():
            logger.warning("LLM nicht verfügbar - fallback zu BM25")
            return await self._bm25_rerank(query, documents, scores)
        
        try:
            # Limitiere auf Top-K Dokumente für LLM
            top_docs = documents[:self.config.max_documents_for_llm]
            top_scores = scores[:self.config.max_documents_for_llm] if scores else None
            
            # LLM Scoring Prompts erstellen
            scoring_prompts = []
            for doc in top_docs:
                prompt = self._create_scoring_prompt(query, doc)
                scoring_prompts.append(prompt)
            
            # Batch-LLM-Inferenz
            if hasattr(self, 'batch_processor') and self.batch_processor:
                result = await self.batch_processor.async_batch_generate(
                    prompts=scoring_prompts,
                    max_tokens=10,
                    temperature=self.config.llm_temperature
                )
                llm_responses = result.outputs
            elif self.llm_handler:
                llm_responses = await self.llm_handler.batch_inference(
                    prompts=scoring_prompts,
                    max_tokens=10,
                    temperature=self.config.llm_temperature
                )
            else:
                raise ValueError("Kein LLM Handler verfügbar")
            
            # Scores aus LLM-Antworten extrahieren
            llm_scores = []
            for response in llm_responses:
                score = self._extract_score_from_response(response)
                llm_scores.append(score)
            
            # Kombiniere mit Original-Scores
            if top_scores:
                combined_scores = [
                    0.7 * llm_score + 0.3 * orig_score
                    for llm_score, orig_score in zip(llm_scores, top_scores)
                ]
            else:
                combined_scores = llm_scores
            
            # Sortieren
            sorted_items = sorted(
                zip(top_docs, combined_scores, range(len(top_docs))),
                key=lambda x: x[1],
                reverse=True
            )
            
            reranked_docs, reranked_scores, original_indices = zip(*sorted_items)
            
            # Verbleibende Dokumente anhängen
            remaining_docs = documents[self.config.max_documents_for_llm:]
            remaining_scores = scores[self.config.max_documents_for_llm:] if scores else [0.1] * len(remaining_docs)
            remaining_indices = list(range(self.config.max_documents_for_llm, len(documents)))
            
            final_docs = list(reranked_docs) + remaining_docs
            final_scores = list(reranked_scores) + remaining_scores
            final_indices = list(original_indices) + remaining_indices
            
            self.stats['llm_reranks'] += 1
            
            return RerankResult(
                documents=final_docs,
                scores=final_scores,
                original_indices=final_indices,
                method_used=RerankMethod.LLM_SCORING,
                processing_time=0.0,
                metadata={
                    'llm_processed_docs': len(top_docs),
                    'total_docs': len(documents),
                    'avg_llm_score': sum(llm_scores) / len(llm_scores) if llm_scores else 0
                }
            )
            
        except Exception as e:
            logger.warning(f"LLM Scoring Reranking fehlgeschlagen: {e}")
            return await self._bm25_rerank(query, documents, scores)
    
    async def _llm_pairwise_rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]]
    ) -> RerankResult:
        """LLM-basiertes Pairwise Comparison Reranking"""
        
        if not self._can_use_llm() or len(documents) < 2:
            return await self._llm_scoring_rerank(query, documents, scores)
        
        try:
            # Für große Listen: Nutze nur Top-K für Pairwise
            top_docs = documents[:min(10, len(documents))]  # Maximal 10 für Pairwise
            
            # Einfache pairwise comparison (vereinfacht)
            comparison_scores = [0] * len(top_docs)
            
            # Erstelle Comparison-Prompts
            comparison_prompts = []
            comparisons = []
            
            for i in range(len(top_docs)):
                for j in range(i + 1, len(top_docs)):
                    prompt = self._create_pairwise_prompt(query, top_docs[i], top_docs[j])
                    comparison_prompts.append(prompt)
                    comparisons.append((i, j))
            
            # Batch-LLM für Comparisons
            if comparison_prompts:
                if hasattr(self, 'batch_processor') and self.batch_processor:
                    result = await self.batch_processor.async_batch_generate(
                        prompts=comparison_prompts,
                        max_tokens=5,
                        temperature=self.config.llm_temperature
                    )
                    comparison_results = result.outputs
                elif self.llm_handler:
                    comparison_results = await self.llm_handler.batch_inference(
                        prompts=comparison_prompts,
                        max_tokens=5,
                        temperature=self.config.llm_temperature
                    )
                else:
                    raise ValueError("Kein LLM Handler verfügbar")
                
                # Scores aus Comparisons berechnen
                for (i, j), result in zip(comparisons, comparison_results):
                    winner = self._extract_winner_from_comparison(result)
                    if winner == 1:  # Erstes Dokument gewinnt
                        comparison_scores[i] += 1
                    elif winner == 2:  # Zweites Dokument gewinnt
                        comparison_scores[j] += 1
            
            # Normalisiere Scores
            max_score = max(comparison_scores) if comparison_scores else 1
            normalized_scores = [score / max_score for score in comparison_scores] if max_score > 0 else comparison_scores
            
            # Sortieren
            sorted_items = sorted(
                zip(top_docs, normalized_scores, range(len(top_docs))),
                key=lambda x: x[1],
                reverse=True
            )
            
            reranked_docs, reranked_scores, original_indices = zip(*sorted_items)
            
            # Verbleibende Dokumente anhängen
            remaining_docs = documents[len(top_docs):]
            remaining_scores = scores[len(top_docs):] if scores else [0.1] * len(remaining_docs)
            remaining_indices = list(range(len(top_docs), len(documents)))
            
            final_docs = list(reranked_docs) + remaining_docs
            final_scores = list(reranked_scores) + remaining_scores
            final_indices = list(original_indices) + remaining_indices
            
            return RerankResult(
                documents=final_docs,
                scores=final_scores,
                original_indices=final_indices,
                method_used=RerankMethod.LLM_PAIRWISE,
                processing_time=0.0,
                metadata={
                    'pairwise_comparisons': len(comparison_prompts),
                    'processed_docs': len(top_docs)
                }
            )
            
        except Exception as e:
            logger.warning(f"LLM Pairwise Reranking fehlgeschlagen: {e}")
            return await self._llm_scoring_rerank(query, documents, scores)
    
    async def _hybrid_rerank(
        self,
        query: str,
        documents: List[str],
        scores: Optional[List[float]]
    ) -> RerankResult:
        """Hybrid-Reranking kombiniert mehrere Methoden"""
        
        try:
            # 1. BM25 Scores
            bm25_result = await self._bm25_rerank(query, documents, scores)
            
            # 2. LLM Scores (falls verfügbar)
            if self._can_use_llm() and len(documents) <= self.config.max_documents_for_llm:
                llm_result = await self._llm_scoring_rerank(query, documents, scores)
                
                # Kombiniere BM25 und LLM Scores
                max_bm25 = max(bm25_result.scores) if bm25_result.scores else 1
                max_llm = max(llm_result.scores) if llm_result.scores else 1
                
                hybrid_scores = []
                for i, (bm25_score, llm_score) in enumerate(zip(bm25_result.scores, llm_result.scores)):
                    # 60% LLM, 40% BM25
                    hybrid_score = 0.6 * (llm_score / max_llm) + 0.4 * (bm25_score / max_bm25)
                    hybrid_scores.append(hybrid_score)
                
                # Diversität hinzufügen
                if self.config.diversity_factor > 0:
                    hybrid_scores = self._add_diversity(documents, hybrid_scores)
                
                # Final sortieren
                sorted_items = sorted(
                    zip(documents, hybrid_scores, range(len(documents))),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                reranked_docs, reranked_scores, original_indices = zip(*sorted_items)
                
                return RerankResult(
                    documents=list(reranked_docs),
                    scores=list(reranked_scores),
                    original_indices=list(original_indices),
                    method_used=RerankMethod.HYBRID,
                    processing_time=0.0,
                    metadata={
                        'bm25_weight': 0.4,
                        'llm_weight': 0.6,
                        'diversity_factor': self.config.diversity_factor,
                        'llm_available': True
                    }
                )
            else:
                # Fallback: Nur BM25 + TF-IDF
                tfidf_result = await self._tfidf_rerank(query, documents, scores)
                
                # Kombiniere BM25 und TF-IDF
                max_bm25 = max(bm25_result.scores) if bm25_result.scores else 1
                max_tfidf = max(tfidf_result.scores) if tfidf_result.scores else 1
                
                hybrid_scores = [
                    0.6 * (bm25 / max_bm25) + 0.4 * (tfidf / max_tfidf)
                    for bm25, tfidf in zip(bm25_result.scores, tfidf_result.scores)
                ]
                
                # Sortieren
                sorted_items = sorted(
                    zip(documents, hybrid_scores, range(len(documents))),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                reranked_docs, reranked_scores, original_indices = zip(*sorted_items)
                
                return RerankResult(
                    documents=list(reranked_docs),
                    scores=list(reranked_scores),
                    original_indices=list(original_indices),
                    method_used=RerankMethod.HYBRID,
                    processing_time=0.0,
                    metadata={
                        'bm25_weight': 0.6,
                        'tfidf_weight': 0.4,
                        'llm_available': False
                    }
                )
                
        except Exception as e:
            logger.warning(f"Hybrid Reranking fehlgeschlagen: {e}")
            return await self._bm25_rerank(query, documents, scores)
    
    # Batch-Processing Methoden
    
    async def _batch_llm_rerank(
        self,
        queries: List[str],
        document_lists: List[List[str]],
        score_lists: Optional[List[List[float]]],
        method: RerankMethod
    ) -> List[RerankResult]:
        """Batch-LLM-Reranking für optimale Performance"""
        
        try:
            # Alle LLM-Prompts sammeln
            all_prompts = []
            prompt_mapping = []  # (query_idx, doc_idx)
            
            for query_idx, (query, documents) in enumerate(zip(queries, document_lists)):
                # Limitiere auf Top-K für LLM
                top_docs = documents[:self.config.max_documents_for_llm]
                
                for doc_idx, doc in enumerate(top_docs):
                    if method == RerankMethod.LLM_SCORING:
                        prompt = self._create_scoring_prompt(query, doc)
                    else:  # LLM_PAIRWISE oder HYBRID
                        prompt = self._create_scoring_prompt(query, doc)
                    
                    all_prompts.append(prompt)
                    prompt_mapping.append((query_idx, doc_idx))
            
            # Batch-LLM-Inferenz
            if hasattr(self, 'batch_processor') and self.batch_processor:
                batch_result = await self.batch_processor.async_batch_generate(
                    prompts=all_prompts,
                    max_tokens=10,
                    temperature=self.config.llm_temperature
                )
                all_responses = batch_result.outputs
            elif self.llm_handler:
                all_responses = await self.llm_handler.batch_inference(
                    prompts=all_prompts,
                    max_tokens=10,
                    temperature=self.config.llm_temperature
                )
            else:
                raise ValueError("Kein LLM Handler verfügbar")
            
            # Responses zu Queries zuordnen
            query_responses = [[] for _ in queries]
            for (query_idx, doc_idx), response in zip(prompt_mapping, all_responses):
                query_responses[query_idx].append((doc_idx, response))
            
            # Ergebnisse für jede Query erstellen
            results = []
            for query_idx, (query, documents) in enumerate(zip(queries, document_lists)):
                scores = score_lists[query_idx] if score_lists else None
                responses = query_responses[query_idx]
                
                # LLM-Scores extrahieren
                llm_scores = [0.0] * len(documents)
                for doc_idx, response in responses:
                    if doc_idx < len(documents):
                        score = self._extract_score_from_response(response)
                        llm_scores[doc_idx] = score
                
                # Kombiniere mit Original-Scores
                if scores:
                    max_llm = max(llm_scores) if any(s > 0 for s in llm_scores) else 1
                    max_orig = max(scores) if scores else 1
                    
                    combined_scores = [
                        0.7 * (llm / max_llm) + 0.3 * (orig / max_orig)
                        for llm, orig in zip(llm_scores, scores)
                    ]
                else:
                    combined_scores = llm_scores
                
                # Sortieren
                sorted_items = sorted(
                    zip(documents, combined_scores, range(len(documents))),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                reranked_docs, reranked_scores, original_indices = zip(*sorted_items)
                
                result = RerankResult(
                    documents=list(reranked_docs),
                    scores=list(reranked_scores),
                    original_indices=list(original_indices),
                    method_used=method,
                    processing_time=0.0,
                    metadata={
                        'llm_responses': len(responses),
                        'batch_processed': True
                    }
                )
                results.append(result)
            
            self.stats['llm_reranks'] += len(queries)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch-LLM-Reranking fehlgeschlagen: {e}")
            # Fallback: Sequentielle Verarbeitung
            return await self._sequential_rerank(queries, document_lists, score_lists, method)
    
    async def _sequential_rerank(
        self,
        queries: List[str],
        document_lists: List[List[str]],
        score_lists: Optional[List[List[float]]],
        method: RerankMethod
    ) -> List[RerankResult]:
        """Sequentielle Verarbeitung als Fallback"""
        
        results = []
        for i, (query, documents) in enumerate(zip(queries, document_lists)):
            scores = score_lists[i] if score_lists else None
            
            try:
                result = await self._execute_reranking(query, documents, scores, method)
                results.append(result)
            except Exception as e:
                logger.warning(f"Sequentielles Reranking für Query {i} fehlgeschlagen: {e}")
                # Fallback: Original-Reihenfolge
                fallback_result = RerankResult(
                    documents=documents,
                    scores=scores or [1.0] * len(documents),
                    original_indices=list(range(len(documents))),
                    method_used=RerankMethod.SIMILARITY_SCORE,
                    processing_time=0.0,
                    metadata={'error': str(e)}
                )
                results.append(fallback_result)
        
        return results
    
    async def _parallel_non_llm_rerank(
        self,
        queries: List[str],
        document_lists: List[List[str]],
        score_lists: Optional[List[List[float]]],
        method: RerankMethod
    ) -> List[RerankResult]:
        """Parallele Verarbeitung für Non-LLM Methoden"""
        
        tasks = []
        for i, (query, documents) in enumerate(zip(queries, document_lists)):
            scores = score_lists[i] if score_lists else None
            task = self._execute_reranking(query, documents, scores, method)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Exception-Handling
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Paralleles Reranking für Query {i} fehlgeschlagen: {result}")
                # Fallback
                documents = document_lists[i]
                scores = score_lists[i] if score_lists else [1.0] * len(documents)
                fallback_result = RerankResult(
                    documents=documents,
                    scores=scores,
                    original_indices=list(range(len(documents))),
                    method_used=RerankMethod.SIMILARITY_SCORE,
                    processing_time=0.0,
                    metadata={'error': str(result)}
                )
                final_results.append(fallback_result)
            else:
                final_results.append(result)
        
        return final_results
    
    # Utility-Methoden
    
    def _can_use_llm(self) -> bool:
        """Prüft ob LLM-Reranking verfügbar ist"""
        return (
            (hasattr(self, 'batch_processor') and self.batch_processor) or
            (self.llm_handler is not None)
        )
    
    def _can_use_batch_llm(self) -> bool:
        """Prüft ob Batch-LLM verfügbar ist"""
        return (
            self.config.enable_batch_llm and
            ((hasattr(self, 'batch_processor') and self.batch_processor) or 
             (self.llm_handler and hasattr(self.llm_handler, 'batch_inference')))
        )
    
    def _create_scoring_prompt(self, query: str, document: str) -> str:
        """Erstellt LLM-Prompt für Dokument-Scoring"""
        
        # Dokumenttext kürzen falls zu lang
        max_doc_length = 500
        if len(document) > max_doc_length:
            document = document[:max_doc_length] + "..."
        
        prompt = f"""Bewerte wie relevant das folgende Dokument für die gegebene Frage ist.

Frage: {query}

Dokument: {document}

Bewertung (0-10, wobei 10 = sehr relevant, 0 = nicht relevant):"""
        
        return prompt
    
    def _create_pairwise_prompt(self, query: str, doc1: str, doc2: str) -> str:
        """Erstellt LLM-Prompt für Pairwise Comparison"""
        
        max_doc_length = 300
        if len(doc1) > max_doc_length:
            doc1 = doc1[:max_doc_length] + "..."
        if len(doc2) > max_doc_length:
            doc2 = doc2[:max_doc_length] + "..."
        
        prompt = f"""Welches Dokument ist relevanter für die gegebene Frage?

Frage: {query}

Dokument A: {doc1}

Dokument B: {doc2}

Antwort (A oder B):"""
        
        return prompt
    
    def _extract_score_from_response(self, response: str) -> float:
        """Extrahiert numerischen Score aus LLM-Response"""
        try:
            # Suche nach Zahlen in der Antwort
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', response)
            
            if numbers:
                score = float(numbers[0])
                # Normalisiere auf 0-1 Range
                if score > 10:
                    score = score / 100  # Falls Prozent angegeben
                elif score > 1:
                    score = score / 10   # Falls 0-10 Scale
                
                return min(1.0, max(0.0, score))
            else:
                # Fallback: Text-basierte Bewertung
                response_lower = response.lower()
                if any(word in response_lower for word in ['sehr relevant', 'hoch', 'excellent', 'perfect']):
                    return 0.9
                elif any(word in response_lower for word in ['relevant', 'gut', 'good']):
                    return 0.7
                elif any(word in response_lower for word in ['teilweise', 'mittel', 'ok']):
                    return 0.5
                elif any(word in response_lower for word in ['wenig', 'niedrig', 'schlecht']):
                    return 0.3
                else:
                    return 0.1
                    
        except Exception as e:
            logger.debug(f"Score-Extraktion fehlgeschlagen: {e}")
            return 0.5  # Neutraler Fallback-Score
    
    def _extract_winner_from_comparison(self, response: str) -> int:
        """Extrahiert Gewinner aus Pairwise Comparison Response"""
        try:
            response_lower = response.lower().strip()
            
            # Suche nach A oder B
            if 'a' in response_lower and 'b' not in response_lower:
                return 1
            elif 'b' in response_lower and 'a' not in response_lower:
                return 2
            elif response_lower.startswith('a'):
                return 1
            elif response_lower.startswith('b'):
                return 2
            else:
                return 0  # Unentschieden
                
        except Exception as e:
            logger.debug(f"Winner-Extraktion fehlgeschlagen: {e}")
            return 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Einfache Tokenisierung für BM25/TF-IDF"""
        # Entferne Satzzeichen und teile an Leerzeichen
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [token for token in tokens if len(token) > 1]  # Filtere sehr kurze Tokens
    
    def _add_diversity(self, documents: List[str], scores: List[float]) -> List[float]:
        """Fügt Diversität zu Scores hinzu um Redundanz zu reduzieren"""
        try:
            if self.config.diversity_factor <= 0:
                return scores
            
            diverse_scores = scores.copy()
            processed_docs = set()
            
            for i, doc in enumerate(documents):
                # Einfache Ähnlichkeitsprüfung basierend auf Worten
                doc_words = set(self._tokenize(doc.lower()))
                
                for processed_doc in processed_docs:
                    processed_words = set(self._tokenize(processed_doc.lower()))
                    
                    # Jaccard-Ähnlichkeit
                    intersection = len(doc_words & processed_words)
                    union = len(doc_words | processed_words)
                    similarity = intersection / union if union > 0 else 0
                    
                    # Reduziere Score basierend auf Ähnlichkeit
                    if similarity > 0.5:
                        diverse_scores[i] *= (1 - self.config.diversity_factor * similarity)
                
                processed_docs.add(doc)
            
            return diverse_scores
            
        except Exception as e:
            logger.debug(f"Diversitäts-Anpassung fehlgeschlagen: {e}")
            return scores
    
    def _generate_cache_key(
        self,
        query: str,
        documents: List[str],
        method: RerankMethod
    ) -> str:
        """Generiert Cache-Key für Reranking"""
        import hashlib
        
        # Erstelle reproduzierbaren Key
        key_components = [
            query,
            str(sorted(documents)),  # Sortiert für Konsistenz
            method.value,
            str(self.config.llm_temperature),
            str(self.config.max_documents_for_llm)
        ]
        
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_stats(self, count: int, processing_time: float, method: RerankMethod):
        """Aktualisiert Performance-Statistiken"""
        self.stats['total_reranks'] += count
        self.stats['method_usage'][method.value] += count
        
        # Gleitender Durchschnitt für Processing Time
        if self.stats['avg_processing_time'] == 0:
            self.stats['avg_processing_time'] = processing_time
        else:
            self.stats['avg_processing_time'] = (
                self.stats['avg_processing_time'] * 0.9 + processing_time * 0.1
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Gibt Performance-Statistiken zurück"""
        
        # Cache Hit Rate
        total_cache_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (
            self.stats['cache_hits'] / total_cache_requests 
            if total_cache_requests > 0 else 0
        )
        
        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.cache) if self.cache else 0,
            'llm_available': self._can_use_llm(),
            'batch_llm_available': self._can_use_batch_llm(),
            'config': {
                'primary_method': self.config.primary_method.value,
                'max_llm_batch_size': self.config.max_llm_batch_size,
                'max_documents_for_llm': self.config.max_documents_for_llm,
                'diversity_factor': self.config.diversity_factor
            }
        }
    
    def clear_cache(self):
        """Leert den Reranking-Cache"""
        if self.cache:
            self.cache.clear()
            self.stats['cache_hits'] = 0
            self.stats['cache_misses'] = 0
            logger.info("Reranking Cache geleert")
    
    async def cleanup(self):
        """Cleanup-Methode"""
        if hasattr(self, 'batch_processor') and self.batch_processor:
            self.batch_processor.cleanup()
        
        if self.cache:
            self.cache.clear()
        
        logger.info("Rerank Engine Cleanup abgeschlossen")


# Utility Functions

async def create_rerank_engine(
    method: RerankMethod = RerankMethod.HYBRID,
    llm_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
    enable_batch_llm: bool = True,
    llm_handler=None
) -> MLXRerankEngine:
    """Factory Function für Rerank Engine"""
    
    config = RerankConfig(
        primary_method=method,
        llm_model=llm_model,
        enable_batch_llm=enable_batch_llm
    )
    
    engine = MLXRerankEngine(config)
    await engine.initialize(llm_handler)
    return engine


async def benchmark_rerank_performance(
    engine: MLXRerankEngine,
    test_queries: Optional[List[str]] = None,
    test_documents: Optional[List[List[str]]] = None
) -> Dict[str, Any]:
    """Benchmark für Reranking-Performance"""
    
    if test_queries is None:
        test_queries = [
            "Was ist Machine Learning?",
            "Wie funktioniert Apple Silicon?",
            "Vorteile von MLX Framework",
            "Performance-Optimierung für Apple Silicon"
        ]
    
    if test_documents is None:
        # Generiere Test-Dokumente
        base_docs = [
            "Machine Learning ist ein Teilbereich der künstlichen Intelligenz.",
            "Apple Silicon nutzt ARM-basierte Prozessoren für bessere Effizienz.",
            "MLX Framework ist optimiert für Apple Silicon Hardware.",
            "Unified Memory Architecture bietet Performance-Vorteile.",
            "Neural Engine beschleunigt Machine Learning Operationen.",
            "Metal Performance Shaders ermöglichen GPU-Computing.",
            "Performance-Optimierung erfordert Hardware-spezifische Anpassungen.",
            "Batch-Processing kann die Durchsatzrate erheblich steigern."
        ]
        
        test_documents = [base_docs for _ in test_queries]
    
    # Single Reranking Benchmark
    single_times = []
    for query, documents in zip(test_queries, test_documents):
        start_time = time.time()
        result = await engine.rerank_with_scores(query, documents)
        single_time = time.time() - start_time
        single_times.append(single_time)
    
    avg_single_time = sum(single_times) / len(single_times)
    
    # Batch Reranking Benchmark
    start_time = time.time()
    batch_result = await engine.batch_rerank(test_queries, test_documents)
    batch_time = time.time() - start_time
    
    return {
        'single_reranking': {
            'avg_time_per_query': avg_single_time,
            'total_time': sum(single_times),
            'queries_count': len(test_queries)
        },
        'batch_reranking': {
            'total_time': batch_time,
            'queries_count': len(test_queries),
            'avg_time_per_query': batch_time / len(test_queries),
            'successful_queries': batch_result.successful_queries,
            'speedup': sum(single_times) / batch_time if batch_time > 0 else 0
        },
        'performance_stats': engine.get_performance_stats()
    }


def get_rerank_method_recommendations(
    document_count: int,
    query_complexity: str = "medium",  # "simple", "medium", "complex"
    llm_available: bool = True
) -> RerankMethod:
    """Empfiehlt optimale Reranking-Methode basierend auf Kontext"""
    
    if document_count <= 5:
        # Wenige Dokumente: LLM-basiert falls verfügbar
        return RerankMethod.LLM_PAIRWISE if llm_available else RerankMethod.BM25
    elif document_count <= 20:
        # Mittlere Anzahl: LLM-Scoring oder Hybrid
        if llm_available:
            return RerankMethod.HYBRID if query_complexity == "complex" else RerankMethod.LLM_SCORING
        else:
            return RerankMethod.BM25
    else:
        # Viele Dokumente: Effiziente Methoden
        if llm_available and query_complexity == "complex":
            return RerankMethod.HYBRID  # LLM nur für Top-K
        else:
            return RerankMethod.BM25 if query_complexity == "simple" else RerankMethod.TFIDF