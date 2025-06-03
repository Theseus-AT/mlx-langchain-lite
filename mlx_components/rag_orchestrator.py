"""
MLX RAG Orchestrator
Koordiniert alle MLX Components für eine komplette RAG Pipeline
Zentraler Hub für Document Processing, Vector Search, Re-ranking und LLM Generation
"""

import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import hashlib
import logging # Added for robust logging
import aiofiles # Added for async file operations in export_configuration

# Configure logging
# You can customize the logging level and format further if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# MLX Components (Assuming these are correctly defined elsewhere)
# from mlx_components.embedding_engine import MLXEmbeddingEngine, EmbeddingConfig, EmbeddingResult
# from mlx_components.vector_store import MLXVectorStore, VectorStoreConfig, QueryResult
# from mlx_components.llm_handler import MLXLLMHandler, LLMConfig, LLMRequest, LLMResponse
# from mlx_components.rerank_engine import MLXRerankEngine, ReRankConfig, RerankResult

# Tools (Assuming these are correctly defined elsewhere)
# from tools.document_processor import MLXDocumentProcessor, ProcessingConfig, ProcessedDocument
# from tools.code_analyzer import MLXCodeAnalyzer, CodeAnalysisConfig, RepositoryAnalysis
# from tools.research_assistant import MLXResearchAssistant, ResearchConfig, ResearchResult

# Placeholder classes if actual imports are not available for linting/running standalone
# These should be replaced by your actual component imports
class MLXEmbeddingEngine:
    def __init__(self, config): self.config = config
    async def initialize(self): logger.info("Mock MLXEmbeddingEngine initialized")
    async def embed(self, texts: List[str]): return EmbeddingResult(embeddings=[[0.1]*10 for _ in texts], token_count=sum(len(t) for t in texts)) # type: ignore
    def get_stats(self): return {"cache_size": 0}
    def clear_cache(self): logger.info("Mock Embedding cache cleared")

@dataclass
class EmbeddingResult:
    embeddings: List[List[float]]
    token_count: int

@dataclass
class EmbeddingConfig:
    model_path: str
    batch_size: int
    cache_embeddings: bool
    normalize_embeddings: bool

class MLXVectorStore:
    def __init__(self, config): self.config = config
    async def initialize(self): logger.info("Mock MLXVectorStore initialized")
    async def query(self, user_id: str, model_id: str, query_vector: List[float], k: int, filters: Optional[Dict], namespace: str):
        return [QueryResult(id=f"doc_{i}", score=0.9-i*0.1, metadata={"text": f"Mock context {i}", "title": f"Mock Title {i}"}) for i in range(k)] # type: ignore
    async def create_user_store(self, user_id: str, model_id: str): logger.info(f"Mock user store created for {user_id}")
    async def get_store_stats(self, user_id: str, model_id: str): return VectorStoreStats(total_vectors=100, storage_size_mb=10, last_updated=datetime.now().isoformat(), query_latency_ms=20) # type: ignore
    async def close(self): logger.info("Mock MLXVectorStore closed")
    def get_performance_stats(self): return {"queries": 10, "avg_latency_ms": 20}


@dataclass
class VectorStoreStats:
    total_vectors: int
    storage_size_mb: float
    last_updated: Optional[str]
    query_latency_ms: float


@dataclass
class QueryResult:
    id: str
    score: float
    metadata: Dict[str, Any]

@dataclass
class VectorStoreConfig:
    base_url: str
    api_key: Optional[str]
    batch_size: int
    default_k: int

class MLXLLMHandler:
    def __init__(self, config): self.config = config
    async def initialize(self): logger.info("Mock MLXLLMHandler initialized")
    async def generate_single(self, request): return LLMResponse(response="Mock LLM single response.", token_count=5) # type: ignore
    async def rag_generate(self, query: str, context_documents: List[Dict], user_id: str, system_prompt: str):
        return LLMResponse(response=f"Mock RAG response for: {query}", token_count=10 + len(context_documents)) # type: ignore
    def get_performance_stats(self): return {"cache_hit_rate": 0.5, "avg_tokens_per_response": 50}
    def clear_cache(self): logger.info("Mock LLM cache cleared")


@dataclass
class LLMConfig:
    model_path: str
    max_tokens: int
    temperature: float
    batch_size: int
    cache_responses: bool

@dataclass
class LLMRequest:
    prompt: str
    user_id: str
    max_tokens: int

@dataclass
class LLMResponse:
    response: str
    token_count: int

class MLXRerankEngine:
    def __init__(self, config): self.config = config
    async def initialize(self): logger.info("Mock MLXRerankEngine initialized")
    async def rerank(self, query: str, candidates: List[Dict], top_k: int):
        # Simple pass-through rerank for mock
        return RerankResult(candidates=sorted(candidates, key=lambda x: x.get('score', 0), reverse=True)[:top_k]) # type: ignore
    def get_performance_stats(self): return {"rerank_queries": 5, "avg_latency_ms": 10}

@dataclass
class ReRankConfig:
    top_k: int
    diversity_factor: float
    enable_diversity_rerank: bool

@dataclass
class RerankResult:
    candidates: List[Any] # Should be List of RerankedCandidate or similar

class MLXDocumentProcessor:
    def __init__(self, config, embedding_engine):
        self.config = config
        self.embedding_engine = embedding_engine
        logger.info("Mock MLXDocumentProcessor initialized")
    async def process_document(self, file_path: Union[str, Path], custom_metadata: Optional[Dict[str, Any]] = None):
        return ProcessedDocument( # type: ignore
            file_name=Path(file_path).name,
            chunks=[ProcessedChunk(text=f"Chunk from {Path(file_path).name}", embedding=[0.1]*10, metadata=custom_metadata or {})], # type: ignore
            summary="Mock document summary."
        )
    async def save_to_vector_store(self, processed_doc, vector_store, user_id: str, model_id: str, namespace: Optional[str] = None):
        logger.info(f"Mock saving {processed_doc.file_name} to vector store for user {user_id} in namespace {namespace or 'default'}.")
        return True # Simulate success
    def get_performance_stats(self): return {"docs_processed": 2, "avg_chunk_time_ms": 50}


@dataclass
class ProcessedChunk:
    text: str
    embedding: List[float]
    metadata: Dict[str, Any]

@dataclass
class ProcessedDocument:
    file_name: str
    chunks: List[ProcessedChunk] # type: ignore
    summary: Optional[str] = None
    # Add other relevant fields like original_metadata, etc.

@dataclass
class ProcessingConfig:
    chunk_size: int
    chunk_overlap: int
    preserve_structure: bool
    auto_summarize: bool
    embedding_model: str # Model path or identifier

class MLXCodeAnalyzer:
    def __init__(self, config, embedding_engine, document_processor):
        self.config = config
        self.embedding_engine = embedding_engine
        self.document_processor = document_processor # Added this based on init
        logger.info("Mock MLXCodeAnalyzer initialized")
    async def analyze_repository(self, repo_path: Union[str, Path], user_id: str):
        return RepositoryAnalysis(project_name=Path(repo_path).name, files_analyzed=5, total_elements=20) # type: ignore
    async def save_to_vector_store(self, analysis, vector_store, user_id: str, model_id: str, namespace: Optional[str] = "code_analysis"):
        logger.info(f"Mock saving code analysis for {analysis.project_name} to vector store for user {user_id} in namespace {namespace}.")
        return True # Simulate success
    def get_performance_stats(self): return {"repos_analyzed": 1, "avg_analysis_time_s": 60}

@dataclass
class RepositoryAnalysis:
    project_name: str
    files_analyzed: int
    total_elements: int
    # Add more fields like language breakdown, complexity scores, embeddings etc.

@dataclass
class CodeAnalysisConfig:
    generate_embeddings: bool
    embedding_model: str
    analyze_complexity: bool

class MLXResearchAssistant:
    def __init__(self, config, embedding_engine, vector_store, llm_handler, rerank_engine, document_processor):
        self.config = config
        # Storing components
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.llm_handler = llm_handler
        self.rerank_engine = rerank_engine
        self.document_processor = document_processor
        logger.info("Mock MLXResearchAssistant initialized")
    async def research(self, search_query):
        return ResearchResult( # type: ignore
            query=search_query.query,
            synthesized_answer=f"Mock research answer for {search_query.query}",
            sources_used=[{"title": "Mock Source", "url": "http://example.com", "domain":"example.com"}],
            confidence_score=0.85,
            follow_up_questions=["Follow up?"],
            related_topics=["Related topic"]
        )
    async def close(self): logger.info("Mock MLXResearchAssistant closed")
    def get_performance_stats(self): return {"research_queries": 3, "avg_sources_per_query": 3}


@dataclass
class SearchQuery: # Added placeholder for ResearchAssistant
    query: str
    user_id: str
    max_results: int
    context: Optional[str] = None

@dataclass
class ResearchConfig:
    auto_summarize: bool
    embedding_model: str
    llm_model: str

@dataclass
class ResearchResult:
    query: str
    synthesized_answer: str
    sources_used: List[Dict[str, Any]]
    confidence_score: float
    follow_up_questions: Optional[List[str]] = None
    related_topics: Optional[List[str]] = None
# End of Placeholder classes

class RAGMode(Enum):
    """RAG Operation Modi"""
    DOCUMENT_QA = "document_qa"        # Standard Document Q&A
    CODE_ASSISTANCE = "code_assistance"  # Code-spezifische Hilfe
    RESEARCH = "research"                # Web Research mit RAG
    CONVERSATIONAL = "conversational"    # Chat mit Kontext
    HYBRID = "hybrid"                    # Kombiniert mehrere Modi

@dataclass
class RAGConfig:
    """Zentrale Konfiguration für RAG Orchestrator"""
    # Model Configurations
    embedding_model: str = "mlx-community/gte-small"
    llm_model: str = "mlx-community/gemma-2-9b-it-4bit"

    # Vector Store
    vector_store_url: str = "http://localhost:8000" # Example, use actual URL
    vector_store_api_key: Optional[str] = None

    # Processing Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_length: int = 8000 # Max characters for LLM context

    # Retrieval Settings
    default_k: int = 5
    max_k: int = 20 # Max results to retrieve for reranking/processing
    rerank_enabled: bool = True
    rerank_top_k: int = 10 # How many results to keep after reranking

    # Generation Settings
    max_tokens: int = 1024 # Max tokens for LLM generation
    temperature: float = 0.1
    batch_size: int = 5 # For LLM batching if supported

    # Performance Settings
    cache_enabled: bool = True
    parallel_processing: bool = True # For batch_query

    # User Management
    enable_user_isolation: bool = True # Conceptual, actual implementation in vector store
    default_namespace: str = "default"

@dataclass
class RAGQuery:
    """RAG Query Definition"""
    query: str
    user_id: str # Essential for multi-tenancy and personalization
    mode: RAGMode = RAGMode.DOCUMENT_QA
    namespace: Optional[str] = None # For scoping data in vector store
    context: Optional[str] = None # Additional context for the query (e.g., previous turn)
    filters: Optional[Dict[str, Any]] = None # Metadata filters for vector search
    k: Optional[int] = None # Number of documents to retrieve
    include_sources: bool = True # Whether to include source documents in response
    conversation_id: Optional[str] = None # For tracking conversational context
    follow_up: bool = False # Indicates if this is a follow-up question

@dataclass
class RAGResponse:
    """RAG Response mit allen Metadaten"""
    query: str
    answer: str
    sources: List[Dict[str, Any]] # List of source documents/snippets
    mode: RAGMode
    user_id: str
    confidence: float # Calculated confidence score for the answer
    processing_time: float # Total time taken to process the query
    token_count: int # Tokens used by LLM for generation
    source_count: int # Number of sources used for the answer
    cache_hit: bool = False
    conversation_id: Optional[str] = None
    follow_up_questions: Optional[List[str]] = None
    related_topics: Optional[List[str]] = None
    debug_info: Optional[Dict[str, Any]] = None # For internal debugging

class MLXRAGOrchestrator:
    """
    High-Performance RAG Orchestrator für MLX Ecosystem

    Features:
    - Unified Interface für alle RAG Operations
    - Multi-Mode Support (Documents, Code, Research, Chat)
    - Intelligent Component Coordination
    - Performance Optimization & Caching
    - User Isolation & Multi-Tenancy (conceptual, depends on vector store)
    - Comprehensive Monitoring (via logging and stats)
    - Brain System Integration Ready (placeholder for future extension)
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        logger.info(f"Initializing MLXRAGOrchestrator with config: {self.config}")

        # Core Components
        self.embedding_engine: Optional[MLXEmbeddingEngine] = None
        self.vector_store: Optional[MLXVectorStore] = None
        self.llm_handler: Optional[MLXLLMHandler] = None
        self.rerank_engine: Optional[MLXRerankEngine] = None

        # Tool Components
        self.document_processor: Optional[MLXDocumentProcessor] = None
        self.code_analyzer: Optional[MLXCodeAnalyzer] = None
        self.research_assistant: Optional[MLXResearchAssistant] = None

        # State Management
        self.initialized = False
        self.component_status: Dict[str, str] = {}

        # Performance Metrics
        self.total_queries = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.mode_usage: Dict[RAGMode, int] = {mode: 0 for mode in RAGMode}

        # Response Cache
        self.response_cache: Optional[Dict[str, RAGResponse]] = {} if self.config.cache_enabled else None
        self.max_cache_size = 1000 # Max items in response cache

        # Active conversations
        self.conversations: Dict[str, List[Dict[str, Any]]] = {} # Stores chat history

    async def initialize(self) -> None:
        """
        Initialisiert alle Components mit optimaler Konfiguration.
        This method is idempotent.
        """
        if self.initialized:
            logger.info("RAG Orchestrator already initialized.")
            return

        logger.info("🚀 Initializing MLX RAG Orchestrator...")
        start_time = time.monotonic()

        try:
            # Initialize Core Components
            await self._initialize_core_components()

            # Initialize Tool Components
            await self._initialize_tool_components()

            # Verify Component Health
            await self._verify_component_health()

            self.initialized = True
            init_time = time.monotonic() - start_time
            logger.info(f"✅ RAG Orchestrator initialized in {init_time:.2f}s")
            self._print_component_status()

        except Exception as e:
            logger.error(f"❌ Failed to initialize RAG Orchestrator: {e}", exc_info=True)
            # Potentially set component statuses to error states
            self.component_status["orchestrator"] = f"❌ Initialization Error: {e}"
            self.initialized = False # Ensure it's marked as not initialized
            raise # Re-raise the exception to signal failure

    async def _initialize_core_components(self) -> None:
        """Initialisiert Core MLX Components"""
        logger.info("Initializing core components...")
        try:
            embedding_config = EmbeddingConfig(
                model_path=self.config.embedding_model,
                batch_size=32, # Component-specific tuning
                cache_embeddings=self.config.cache_enabled,
                normalize_embeddings=True
            )
            self.embedding_engine = MLXEmbeddingEngine(embedding_config)
            await self.embedding_engine.initialize()
            self.component_status["embedding_engine"] = "✅ Ready"
        except Exception as e:
            self.component_status["embedding_engine"] = f"❌ Error: {e}"
            logger.error(f"Failed to initialize Embedding Engine: {e}", exc_info=True)
            raise

        try:
            vector_config = VectorStoreConfig(
                base_url=self.config.vector_store_url,
                api_key=self.config.vector_store_api_key,
                batch_size=100, # Component-specific tuning
                default_k=self.config.default_k
            )
            self.vector_store = MLXVectorStore(vector_config)
            await self.vector_store.initialize()
            self.component_status["vector_store"] = "✅ Ready"
        except Exception as e:
            self.component_status["vector_store"] = f"❌ Error: {e}"
            logger.error(f"Failed to initialize Vector Store: {e}", exc_info=True)
            raise

        try:
            llm_config = LLMConfig(
                model_path=self.config.llm_model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                batch_size=self.config.batch_size,
                cache_responses=self.config.cache_enabled
            )
            self.llm_handler = MLXLLMHandler(llm_config)
            await self.llm_handler.initialize()
            self.component_status["llm_handler"] = "✅ Ready"
        except Exception as e:
            self.component_status["llm_handler"] = f"❌ Error: {e}"
            logger.error(f"Failed to initialize LLM Handler: {e}", exc_info=True)
            raise

        if self.config.rerank_enabled:
            try:
                rerank_config = ReRankConfig(
                    top_k=self.config.rerank_top_k,
                    diversity_factor=0.3, # Example, could be configurable
                    enable_diversity_rerank=True # Example
                )
                self.rerank_engine = MLXRerankEngine(rerank_config)
                await self.rerank_engine.initialize()
                self.component_status["rerank_engine"] = "✅ Ready"
            except Exception as e:
                self.component_status["rerank_engine"] = f"❌ Error: {e}"
                logger.error(f"Failed to initialize Re-rank Engine: {e}", exc_info=True)
                # Continue without reranking if it fails, but log it
                self.rerank_engine = None # Ensure it's None if init fails
        else:
            self.component_status["rerank_engine"] = "⚠️ Disabled by config"
            logger.info("Re-rank engine is disabled by configuration.")

    async def _initialize_tool_components(self) -> None:
        """Initialisiert Tool Components"""
        logger.info("Initializing tool components...")
        # Ensure core components are available for tools that depend on them
        if not self.embedding_engine:
            logger.error("Cannot initialize tool components: Embedding Engine is missing.")
            # Set status for tools that depend on it
            self.component_status["document_processor"] = "❌ Error: Embedding Engine missing"
            self.component_status["code_analyzer"] = "❌ Error: Embedding Engine missing"
            self.component_status["research_assistant"] = "❌ Error: Core components missing"
            return

        try:
            doc_config = ProcessingConfig(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                preserve_structure=True,
                auto_summarize=True, # Example, could be configurable
                embedding_model=self.config.embedding_model
            )
            self.document_processor = MLXDocumentProcessor(doc_config, self.embedding_engine)
            self.component_status["document_processor"] = "✅ Ready"
        except Exception as e:
            self.component_status["document_processor"] = f"❌ Error: {e}"
            logger.error(f"Failed to initialize Document Processor: {e}", exc_info=True)
            # No raise here, other tools might still initialize

        if not self.document_processor:
             logger.warning("Code Analyzer depends on Document Processor, which failed to initialize.")
             self.component_status["code_analyzer"] = "❌ Error: Document Processor missing"
        else:
            try:
                code_config = CodeAnalysisConfig(
                    generate_embeddings=True,
                    embedding_model=self.config.embedding_model,
                    analyze_complexity=True # Example
                )
                self.code_analyzer = MLXCodeAnalyzer(code_config, self.embedding_engine, self.document_processor)
                self.component_status["code_analyzer"] = "✅ Ready"
            except Exception as e:
                self.component_status["code_analyzer"] = f"❌ Error: {e}"
                logger.error(f"Failed to initialize Code Analyzer: {e}", exc_info=True)


        # Research assistant depends on multiple components
        if not (self.vector_store and self.llm_handler):
            logger.error("Cannot initialize Research Assistant: Vector Store or LLM Handler is missing.")
            self.component_status["research_assistant"] = "❌ Error: Core components missing"
            return
        try:
            research_config = ResearchConfig(
                auto_summarize=True,
                embedding_model=self.config.embedding_model,
                llm_model=self.config.llm_model
            )
            self.research_assistant = MLXResearchAssistant(
                research_config,
                self.embedding_engine,
                self.vector_store,
                self.llm_handler,
                self.rerank_engine, # Can be None if disabled/failed
                self.document_processor # Can be None if failed
            )
            self.component_status["research_assistant"] = "✅ Ready"
        except Exception as e:
            self.component_status["research_assistant"] = f"❌ Error: {e}"
            logger.error(f"Failed to initialize Research Assistant: {e}", exc_info=True)


    async def _verify_component_health(self) -> None:
        """Überprüft Health aller initialisierten Components durch Testaufrufe."""
        logger.info("Verifying component health...")

        if self.embedding_engine:
            try:
                test_embedding = await self.embedding_engine.embed(["health check"])
                if test_embedding and test_embedding.embeddings:
                    self.component_status["embedding_engine"] += " (Tested)"
                else:
                    self.component_status["embedding_engine"] = "⚠️ Test failed (no embeddings)"
            except Exception as e:
                self.component_status["embedding_engine"] = f"❌ Test Error: {e}"
                logger.warning(f"Embedding engine health check failed: {e}", exc_info=True)

        if self.vector_store:
            try:
                # Use a unique ID for health check to avoid conflicts
                health_check_user_id = f"health_check_{int(time.time())}"
                await self.vector_store.create_user_store(health_check_user_id, "test_model_health")
                self.component_status["vector_store"] += " (Connected & Writable)"
                # Consider deleting the test store if possible/needed
            except Exception as e:
                self.component_status["vector_store"] = f"⚠️ Connection/Write issue: {e}"
                logger.warning(f"Vector store health check failed: {e}", exc_info=True)

        if self.llm_handler:
            try:
                test_request = LLMRequest(
                    prompt="Hello",
                    user_id="health_check_llm",
                    max_tokens=10
                )
                test_response = await self.llm_handler.generate_single(test_request)
                if test_response and test_response.response:
                    self.component_status["llm_handler"] += " (Tested)"
                else:
                    self.component_status["llm_handler"] = "⚠️ Test failed (no response)"
            except Exception as e:
                self.component_status["llm_handler"] = f"❌ Test Error: {e}"
                logger.warning(f"LLM handler health check failed: {e}", exc_info=True)
        # Note: Tool components are generally not tested here as their health depends on core components.

    def _print_component_status(self) -> None:
        """Gibt Component Status über Logging aus."""
        logger.info("\n📊 Component Status:")
        for component, status in self.component_status.items():
            logger.info(f"  {component:<25}: {status}")

    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Hauptfunktion: Verarbeitet RAG Query mit automatischer Mode-Detection.
        Stellt sicher, dass das System initialisiert ist, bevor eine Query verarbeitet wird.
        """
        start_time_mono = time.monotonic() # More precise for performance measurement
        logger.info(f"Received query: '{rag_query.query}' for user '{rag_query.user_id}' in mode '{rag_query.mode.value}'")

        if not self.initialized:
            logger.warning("Orchestrator not initialized. Attempting to initialize now.")
            try:
                await self.initialize()
                if not self.initialized: # Check again after attempt
                    logger.error("Initialization failed. Cannot process query.")
                    # Construct a meaningful error response
                    return RAGResponse(
                        query=rag_query.query,
                        answer="Error: RAG system is not initialized.",
                        sources=[],
                        mode=rag_query.mode,
                        user_id=rag_query.user_id,
                        confidence=0.0,
                        processing_time=time.monotonic() - start_time_mono,
                        token_count=0,
                        source_count=0,
                        conversation_id=rag_query.conversation_id,
                        debug_info={"error": "System not initialized"}
                    )
            except Exception as init_error:
                 logger.error(f"Critical error during on-demand initialization: {init_error}", exc_info=True)
                 return RAGResponse(
                        query=rag_query.query,
                        answer=f"Error: RAG system failed to initialize on demand: {init_error}",
                        sources=[],
                        mode=rag_query.mode,
                        user_id=rag_query.user_id,
                        confidence=0.0,
                        processing_time=time.monotonic() - start_time_mono,
                        token_count=0,
                        source_count=0,
                        conversation_id=rag_query.conversation_id,
                        debug_info={"error": f"On-demand initialization failed: {init_error}"}
                    )


        # Check cache first
        cache_key = self._get_cache_key(rag_query)
        if self.response_cache is not None and cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            cached_response.cache_hit = True
            cached_response.processing_time = time.monotonic() - start_time_mono # Update with actual time
            self.cache_hits += 1
            logger.info(f"Cache hit for query: '{rag_query.query}'. Returning cached response.")
            return cached_response

        try:
            response: RAGResponse
            # Route to appropriate handler based on mode
            if rag_query.mode == RAGMode.DOCUMENT_QA:
                response = await self._handle_document_qa(rag_query)
            elif rag_query.mode == RAGMode.CODE_ASSISTANCE:
                response = await self._handle_code_assistance(rag_query)
            elif rag_query.mode == RAGMode.RESEARCH:
                response = await self._handle_research(rag_query)
            elif rag_query.mode == RAGMode.CONVERSATIONAL:
                response = await self._handle_conversational(rag_query)
            elif rag_query.mode == RAGMode.HYBRID:
                response = await self._handle_hybrid(rag_query)
            else:
                # This case should ideally not be reached if RAGMode is used correctly
                logger.error(f"Unsupported RAG mode: {rag_query.mode}")
                raise ValueError(f"Unsupported RAG mode: {rag_query.mode}")

            # Calculate processing time (already set by handlers, but ensure it's final)
            response.processing_time = time.monotonic() - start_time_mono

            # Update metrics
            self.total_queries += 1
            self.total_processing_time += response.processing_time
            self.mode_usage[rag_query.mode] += 1
            self.cache_misses += 1

            # Cache response
            if self.response_cache is not None:
                self._update_cache(cache_key, response)

            logger.info(f"✅ RAG Query processed in {response.processing_time:.3f}s (mode: {rag_query.mode.value})")
            return response

        except Exception as e:
            logger.error(f"❌ RAG Query failed for '{rag_query.query}': {e}", exc_info=True)
            # Return error response
            return RAGResponse(
                query=rag_query.query,
                answer=f"Error processing query: {str(e)}",
                sources=[],
                mode=rag_query.mode,
                user_id=rag_query.user_id,
                confidence=0.0,
                processing_time=time.monotonic() - start_time_mono,
                token_count=0,
                source_count=0,
                conversation_id=rag_query.conversation_id,
                debug_info={"error": str(e), "traceback": traceback.format_exc() if hasattr(traceback, 'format_exc') else None}
            )

    async def _handle_document_qa(self, rag_query: RAGQuery) -> RAGResponse:
        """Standard Document Q&A Pipeline"""
        start_time_mono = time.monotonic()
        if not self.embedding_engine or not self.vector_store or not self.llm_handler:
            logger.error("Core components (Embedding, VectorStore, LLM) not available for Document Q&A.")
            raise RuntimeError("Core components not initialized for Document Q&A.")

        # 1. Generate query embedding
        query_embedding_result = await self.embedding_engine.embed([rag_query.query])
        if not query_embedding_result.embeddings:
            logger.error("Failed to generate query embedding.")
            raise ValueError("Query embedding generation failed.")

        # 2. Vector search
        k_retrieval = rag_query.k or self.config.default_k
        # Retrieve more for reranking if enabled, up to max_k
        k_for_search = min(k_retrieval * 2 if self.rerank_engine else k_retrieval, self.config.max_k)
        namespace = rag_query.namespace or self.config.default_namespace

        search_results_raw = await self.vector_store.query(
            user_id=rag_query.user_id,
            model_id=self.config.embedding_model.split('/')[-1], # Assuming model_id is suffix
            query_vector=query_embedding_result.embeddings[0],
            k=k_for_search,
            filters=rag_query.filters,
            namespace=namespace
        )

        # 3. Re-ranking (if enabled and results exist)
        final_results_retrieved: List[QueryResult]
        if self.rerank_engine and self.config.rerank_enabled and search_results_raw:
            candidates = [
                {
                    "id": result.id,
                    "content": result.metadata.get("text", ""), # Ensure content key matches reranker expectation
                    "metadata": result.metadata,
                    "score": result.score # Original retrieval score
                } for result in search_results_raw
            ]
            try:
                rerank_output = await self.rerank_engine.rerank(
                    query=rag_query.query,
                    candidates=candidates,
                    top_k=k_retrieval # Rerank to the desired final k
                )
                # Map reranked candidates back to original QueryResult objects or structure
                reranked_ids_scores = {candidate.get('id'): candidate.get('new_score', candidate.get('score')) for candidate in rerank_output.candidates}
                
                # Preserve original QueryResult objects but update order and potentially score
                temp_results_map = {res.id: res for res in search_results_raw}
                final_results_retrieved = []
                for cand in rerank_output.candidates:
                    original_res = temp_results_map.get(cand.get('id'))
                    if original_res:
                        # Optionally update score if reranker provides a new one
                        # original_res.score = reranked_ids_scores.get(original_res.id, original_res.score)
                        final_results_retrieved.append(original_res)

                logger.info(f"Reranked {len(search_results_raw)} candidates to {len(final_results_retrieved)} for query '{rag_query.query}'.")
            except Exception as rerank_err:
                logger.warning(f"Reranking failed: {rerank_err}. Using original search results.", exc_info=True)
                final_results_retrieved = search_results_raw[:k_retrieval]
        else:
            final_results_retrieved = search_results_raw[:k_retrieval]

        # 4. Build context for LLM
        context_docs_for_llm = []
        current_context_length = 0
        for result in final_results_retrieved:
            content = result.metadata.get("text", "")
            if current_context_length + len(content) > self.config.max_context_length and context_docs_for_llm:
                logger.warning(f"Context length limit ({self.config.max_context_length} chars) reached. Truncating context.")
                break
            context_docs_for_llm.append({
                "content": content,
                "metadata": result.metadata # Pass along all metadata
            })
            current_context_length += len(content)

        # 5. Generate response using LLM
        system_prompt = "You are a helpful AI assistant. Answer questions based on the provided context. Cite sources if applicable using their titles or IDs."
        llm_response_obj = await self.llm_handler.rag_generate(
            query=rag_query.query,
            context_documents=context_docs_for_llm,
            user_id=rag_query.user_id,
            system_prompt=system_prompt
        )

        # 6. Prepare sources for RAGResponse
        sources_for_response = []
        if rag_query.include_sources:
            for result in final_results_retrieved: # Use the (potentially reranked) final results
                source_info = {
                    "id": result.id,
                    "title": result.metadata.get("title", result.metadata.get("filename", "Unknown Source")),
                    "content_preview": result.metadata.get("text", "")[:250] + "...", # Preview
                    "score": result.score, # Retrieval score
                    "metadata": result.metadata # Full metadata
                }
                sources_for_response.append(source_info)
        
        processing_time = time.monotonic() - start_time_mono
        return RAGResponse(
            query=rag_query.query,
            answer=llm_response_obj.response,
            sources=sources_for_response,
            mode=rag_query.mode,
            user_id=rag_query.user_id,
            confidence=self._calculate_confidence(final_results_retrieved, llm_response_obj),
            processing_time=processing_time,
            token_count=llm_response_obj.token_count,
            source_count=len(final_results_retrieved),
            conversation_id=rag_query.conversation_id
        )

    async def _handle_code_assistance(self, rag_query: RAGQuery) -> RAGResponse:
        """Code-spezifische Assistance Pipeline"""
        start_time_mono = time.monotonic()
        if not self.embedding_engine or not self.vector_store or not self.llm_handler:
            logger.error("Core components not available for Code Assistance.")
            raise RuntimeError("Core components not initialized for Code Assistance.")

        query_embedding_result = await self.embedding_engine.embed([rag_query.query])
        if not query_embedding_result.embeddings:
            raise ValueError("Query embedding generation failed for code assistance.")

        # Search in code-specific namespace or use filters
        namespace = rag_query.namespace or "code_analysis" # Default namespace for code
        k_retrieval = rag_query.k or self.config.default_k

        # Default filters for code, can be overridden/extended by rag_query.filters
        code_filters = {"type": "code_element"}
        if rag_query.filters:
            code_filters.update(rag_query.filters)

        search_results_raw = await self.vector_store.query(
            user_id=rag_query.user_id,
            model_id=self.config.embedding_model.split('/')[-1],
            query_vector=query_embedding_result.embeddings[0],
            k=k_retrieval, # Reranking might be less common or different for code
            filters=code_filters,
            namespace=namespace
        )
        # Note: Reranking for code might need a specialized model or logic.
        # For simplicity, direct results are used here. Add reranking if a suitable code reranker is available.
        final_results_retrieved = search_results_raw # Or apply reranking if configured

        context_docs_for_llm = []
        current_context_length = 0
        for result in final_results_retrieved:
            content = result.metadata.get("text", result.metadata.get("code_snippet", ""))
            if current_context_length + len(content) > self.config.max_context_length and context_docs_for_llm:
                break
            context_docs_for_llm.append({
                "content": content,
                "metadata": result.metadata
            })
            current_context_length += len(content)
        
        system_prompt = """You are an expert AI code assistant.
Use the provided code context to answer questions, explain functionality, suggest improvements, or generate code snippets.
Be precise and refer to specific functions, classes, or file paths from the context if relevant."""
        llm_response_obj = await self.llm_handler.rag_generate(
            query=rag_query.query,
            context_documents=context_docs_for_llm,
            user_id=rag_query.user_id,
            system_prompt=system_prompt
        )

        sources_for_response = []
        if rag_query.include_sources:
            for result in final_results_retrieved:
                sources_for_response.append({
                    "id": result.id,
                    "title": result.metadata.get("element_name", result.metadata.get("file_path", "Code Snippet")),
                    "file_path": result.metadata.get("file_path", "N/A"),
                    "language": result.metadata.get("language", "N/A"),
                    "content_preview": result.metadata.get("text", "")[:300] + "...",
                    "score": result.score,
                    "metadata": result.metadata
                })
        
        processing_time = time.monotonic() - start_time_mono
        return RAGResponse(
            query=rag_query.query,
            answer=llm_response_obj.response,
            sources=sources_for_response,
            mode=rag_query.mode,
            user_id=rag_query.user_id,
            confidence=self._calculate_confidence(final_results_retrieved, llm_response_obj),
            processing_time=processing_time,
            token_count=llm_response_obj.token_count,
            source_count=len(final_results_retrieved),
            conversation_id=rag_query.conversation_id
        )

    async def _handle_research(self, rag_query: RAGQuery) -> RAGResponse:
        """Web Research Pipeline mit RAG (delegiert an ResearchAssistant)"""
        start_time_mono = time.monotonic()
        if not self.research_assistant:
            logger.error("Research Assistant component not available.")
            raise RuntimeError("Research Assistant not initialized.")

        # Adapt RAGQuery to SearchQuery for the assistant
        search_query_for_assistant = SearchQuery(
            query=rag_query.query,
            user_id=rag_query.user_id,
            max_results=rag_query.k or 10, # Default to 10 results for research
            context=rag_query.context
        )

        research_result_obj = await self.research_assistant.research(search_query_for_assistant)

        sources_for_response = []
        if rag_query.include_sources and research_result_obj.sources_used:
            for source in research_result_obj.sources_used:
                sources_for_response.append({
                    "id": source.get("url", source.get("title", "Research Source")), # Use URL as ID if available
                    "title": source.get("title", "Unknown Web Source"),
                    "url": source.get("url"),
                    "domain": source.get("domain"),
                    "content_preview": source.get("snippet", "")[:250] + "...",
                    "type": "web_research", # Mark source type
                    "metadata": source # Store original source dict as metadata
                })
        
        processing_time = time.monotonic() - start_time_mono
        return RAGResponse(
            query=rag_query.query,
            answer=research_result_obj.synthesized_answer,
            sources=sources_for_response,
            mode=rag_query.mode,
            user_id=rag_query.user_id,
            confidence=research_result_obj.confidence_score, # Assuming assistant provides this
            processing_time=processing_time,
            token_count=len(research_result_obj.synthesized_answer.split()), # Approximate token count
            source_count=len(research_result_obj.sources_used or []),
            follow_up_questions=research_result_obj.follow_up_questions,
            related_topics=research_result_obj.related_topics,
            conversation_id=rag_query.conversation_id
        )

    async def _handle_conversational(self, rag_query: RAGQuery) -> RAGResponse:
        """Conversational RAG mit Chat History"""
        start_time_mono = time.monotonic()
        if not self.embedding_engine or not self.vector_store or not self.llm_handler:
            logger.error("Core components not available for Conversational RAG.")
            raise RuntimeError("Core components not initialized for Conversational RAG.")

        # Get or create conversation ID
        conversation_id = rag_query.conversation_id or f"conv_{rag_query.user_id}_{int(time.time())}"
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        conversation_history = self.conversations[conversation_id]

        # Build conversational context string for LLM and potentially for embedding
        history_for_prompt = []
        # Take last N turns for the prompt context (e.g., last 3 pairs)
        for turn in conversation_history[-3:]:
            history_for_prompt.append(f"Human: {turn['query']}")
            history_for_prompt.append(f"Assistant: {turn['response']}")
        
        # Text to embed: current query + recent history summary or just current query
        # A more advanced approach might embed a condensed version of the conversation.
        # For simplicity, embedding the current query, optionally prefixed by context.
        text_to_embed = rag_query.query
        if rag_query.context: # If explicit context is provided
             text_to_embed = f"{rag_query.context}\nQuestion: {rag_query.query}"
        elif history_for_prompt: # Or use recent chat history
             condensed_history = "\n".join(history_for_prompt[-2:]) # Last user query and AI response
             text_to_embed = f"Previous turn:\n{condensed_history}\n\nCurrent question: {rag_query.query}"


        query_embedding_result = await self.embedding_engine.embed([text_to_embed])
        if not query_embedding_result.embeddings:
            raise ValueError("Query embedding generation failed for conversational RAG.")

        k_retrieval = rag_query.k or self.config.default_k
        namespace = rag_query.namespace or self.config.default_namespace
        search_results_raw = await self.vector_store.query(
            user_id=rag_query.user_id,
            model_id=self.config.embedding_model.split('/')[-1],
            query_vector=query_embedding_result.embeddings[0],
            k=k_retrieval, # Consider if reranking is useful here
            filters=rag_query.filters,
            namespace=namespace
        )
        final_results_retrieved = search_results_raw # Add reranking if beneficial

        context_docs_for_llm = []
        current_context_length = 0
        for result in final_results_retrieved:
            content = result.metadata.get("text", "")
            if current_context_length + len(content) > self.config.max_context_length and context_docs_for_llm:
                break
            context_docs_for_llm.append({
                "content": content,
                "metadata": result.metadata
            })
            current_context_length += len(content)

        # Construct system prompt with conversation history
        conversation_history_str = "\n".join(history_for_prompt)
        system_prompt = f"""You are a helpful AI assistant engaged in a conversation.
Use the conversation history and any retrieved context to provide a relevant and natural response.

Conversation History:
{conversation_history_str}

Retrieved Context (if any) will be provided before your turn.
Answer the current human question: {rag_query.query}"""

        llm_response_obj = await self.llm_handler.rag_generate(
            query=rag_query.query, # The LLM still needs the raw current query
            context_documents=context_docs_for_llm,
            user_id=rag_query.user_id,
            system_prompt=system_prompt
        )

        # Update conversation history
        self.conversations[conversation_id].append({
            "query": rag_query.query,
            "response": llm_response_obj.response,
            "timestamp": datetime.now().isoformat(),
            "mode": rag_query.mode.value
        })
        # Limit conversation history length
        if len(self.conversations[conversation_id]) > 10: # Keep last 10 turns
            self.conversations[conversation_id] = self.conversations[conversation_id][-10:]

        sources_for_response = []
        if rag_query.include_sources:
            for result in final_results_retrieved:
                sources_for_response.append({
                    "id": result.id,
                    "title": result.metadata.get("title", "Context Document"),
                    "content_preview": result.metadata.get("text", "")[:200] + "...",
                    "score": result.score,
                    "metadata": result.metadata
                })
        
        processing_time = time.monotonic() - start_time_mono
        return RAGResponse(
            query=rag_query.query,
            answer=llm_response_obj.response,
            sources=sources_for_response,
            mode=rag_query.mode,
            user_id=rag_query.user_id,
            confidence=self._calculate_confidence(final_results_retrieved, llm_response_obj),
            processing_time=processing_time,
            token_count=llm_response_obj.token_count,
            source_count=len(final_results_retrieved),
            conversation_id=conversation_id # Return the used/generated conversation ID
        )

    async def _handle_hybrid(self, rag_query: RAGQuery) -> RAGResponse:
        """Hybrid Pipeline - kombiniert multiple Approaches und synthetisiert."""
        start_time_mono = time.monotonic()
        logger.info(f"Handling hybrid query: {rag_query.query}")

        # Define sub-queries/tasks for different modes
        tasks_to_run = []
        # Always include document Q&A
        doc_qa_sub_query = RAGQuery(
            query=rag_query.query, user_id=rag_query.user_id, mode=RAGMode.DOCUMENT_QA,
            namespace=rag_query.namespace or self.config.default_namespace, # Use specific or default
            filters=rag_query.filters, k=(rag_query.k or self.config.default_k) // 2 or 1, # Get fewer from each source
            include_sources=True # Need sources for synthesis
        )
        tasks_to_run.append(self._handle_document_qa(doc_qa_sub_query))

        # Conditionally add code assistance if query seems code-related
        code_keywords = ["code", "function", "class", "method", "python", "javascript", "java", "debug", "algorithm"]
        if any(keyword in rag_query.query.lower() for keyword in code_keywords) and self.code_analyzer:
            code_sub_query = RAGQuery(
                query=rag_query.query, user_id=rag_query.user_id, mode=RAGMode.CODE_ASSISTANCE,
                namespace="code_analysis", # Specific namespace for code
                k=(rag_query.k or self.config.default_k) // 2 or 1,
                include_sources=True
            )
            tasks_to_run.append(self._handle_code_assistance(code_sub_query))
        
        # Conditionally add research if query seems to require web knowledge and assistant is available
        research_keywords = ["what is", "latest", "trends", "news", "who is", "explain"]
        if any(keyword in rag_query.query.lower() for keyword in research_keywords) and self.research_assistant:
             research_sub_query = RAGQuery(
                query=rag_query.query, user_id=rag_query.user_id, mode=RAGMode.RESEARCH,
                k=(rag_query.k or self.config.default_k) // 2 or 1, # Research assistant might use its own k
                include_sources=True
            )
             tasks_to_run.append(self._handle_research(research_sub_query))


        # Execute tasks in parallel
        # return_exceptions=True allows us to handle failures gracefully
        results_from_handlers = await asyncio.gather(*tasks_to_run, return_exceptions=True)

        combined_context_parts = []
        all_sources_for_response = []
        total_source_count_intermediate = 0

        for i, result_or_exc in enumerate(results_from_handlers):
            original_task_query = tasks_to_run[i] # This is a coroutine, need to get RAGQuery from it if possible
            # This is a bit tricky, ideally we'd map tasks to their original RAGQuery objects
            # For now, we assume the mode from the result is sufficient
            
            if isinstance(result_or_exc, RAGResponse):
                response_from_handler = result_or_exc
                logger.info(f"Hybrid: Got response from mode {response_from_handler.mode.value} with {response_from_handler.source_count} sources.")
                # Add a prefix to the answer to indicate its origin for the synthesis prompt
                if response_from_handler.answer and not response_from_handler.answer.startswith("Error:"):
                    combined_context_parts.append(f"Information from {response_from_handler.mode.name} search:\n{response_from_handler.answer}")
                
                if rag_query.include_sources:
                    for source in response_from_handler.sources:
                        source["original_mode"] = response_from_handler.mode.value # Tag source origin
                        all_sources_for_response.append(source)
                total_source_count_intermediate += response_from_handler.source_count
            elif isinstance(result_or_exc, Exception):
                logger.warning(f"Hybrid: Sub-task failed: {result_or_exc}", exc_info=True)
                # Optionally add error information to combined_context_parts or debug_info
                combined_context_parts.append(f"Note: A sub-search failed with error: {str(result_or_exc)[:100]}...")


        if not combined_context_parts:
            logger.warning("Hybrid query yielded no usable information from sub-handlers.")
            final_answer = "Could not gather sufficient information from available sources to answer the hybrid query."
            final_token_count = 0
        else:
            synthesis_context_str = "\n\n---\n\n".join(combined_context_parts)
            synthesis_prompt = f"""You are a meta-assistant. Your task is to synthesize a comprehensive answer to the user's query based on information gathered from multiple specialized searches.

User's Original Query: {rag_query.query}

Information Gathered:
{synthesis_context_str}

---
Based on all the above information, provide a single, coherent, and well-structured answer to the user's original query.
If there are conflicting pieces of information, try to reconcile them or state the different perspectives.
Do not just list the information; synthesize it into a new, helpful response.
"""
            if not self.llm_handler:
                raise RuntimeError("LLM Handler not available for hybrid synthesis.")

            synthesis_llm_request = LLMRequest(
                prompt=synthesis_prompt,
                user_id=rag_query.user_id,
                max_tokens=self.config.max_tokens # Allow ample tokens for synthesis
            )
            synthesis_llm_response = await self.llm_handler.generate_single(synthesis_llm_request)
            final_answer = synthesis_llm_response.response
            final_token_count = synthesis_llm_response.token_count
        
        # Calculate combined confidence (simple average for now)
        # More sophisticated methods could weigh by source type or sub-query confidence
        avg_confidence = 0.0
        valid_confidences = [res.confidence for res in results_from_handlers if isinstance(res, RAGResponse) and res.confidence > 0]
        if valid_confidences:
            avg_confidence = sum(valid_confidences) / len(valid_confidences)
        
        processing_time = time.monotonic() - start_time_mono
        return RAGResponse(
            query=rag_query.query,
            answer=final_answer,
            sources=all_sources_for_response,
            mode=rag_query.mode,
            user_id=rag_query.user_id,
            confidence=avg_confidence,
            processing_time=processing_time,
            token_count=final_token_count,
            source_count=len(all_sources_for_response), # Count unique sources if IDs are reliable
            conversation_id=rag_query.conversation_id,
            debug_info={"intermediate_results_count": len(results_from_handlers)}
        )

    async def add_document(self,
                           file_path: Union[str, Path],
                           user_id: str,
                           namespace: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Fügt Dokument zum RAG System hinzu. Verarbeitet und speichert es im Vektor Store.
        """
        if not self.initialized:
            logger.warning("Orchestrator not initialized. Attempting to initialize for add_document.")
            await self.initialize()
            if not self.initialized:
                 logger.error("Initialization failed. Cannot add document.")
                 return False


        if not self.document_processor:
            logger.error("Document Processor component is not available. Cannot add document.")
            return False
        if not self.vector_store:
            logger.error("Vector Store component is not available. Cannot add document.")
            return False

        logger.info(f"Attempting to add document: {file_path} for user: {user_id}, namespace: {namespace or self.config.default_namespace}")
        try:
            # Process document
            # The custom_metadata in process_document is for the document itself,
            # which might then be propagated to chunks by the processor.
            processed_doc_obj = await self.document_processor.process_document(
                file_path=file_path,
                custom_metadata=metadata # Pass document-level metadata
            )

            if not processed_doc_obj or not processed_doc_obj.chunks:
                logger.warning(f"Document processing for '{file_path}' yielded no processable chunks.")
                return False

            # Save to vector store
            namespace_to_use = namespace or self.config.default_namespace
            model_id_suffix = self.config.embedding_model.split('/')[-1]

            # Assuming save_to_vector_store is part of document_processor
            # and it handles iterating through chunks and saving them.
            success = await self.document_processor.save_to_vector_store(
                processed_doc=processed_doc_obj,
                vector_store=self.vector_store,
                user_id=user_id,
                model_id=model_id_suffix, # This should be the ID of the embedding model used
                namespace=namespace_to_use
            )

            if success:
                logger.info(f"✅ Document '{Path(file_path).name}' added successfully to namespace '{namespace_to_use}'.")
            else:
                logger.warning(f"⚠️ Failed to save document '{Path(file_path).name}' to vector store.")
            return success

        except Exception as e:
            logger.error(f"❌ Failed to add document '{file_path}': {e}", exc_info=True)
            return False

    async def add_code_repository(self,
                                  repo_path: Union[str, Path],
                                  user_id: str,
                                  namespace: Optional[str] = "code_analysis") -> bool: # Default namespace for code
        """
        Analysiert und fügt Code Repository zum RAG System hinzu.
        """
        if not self.initialized:
            logger.warning("Orchestrator not initialized. Attempting to initialize for add_code_repository.")
            await self.initialize()
            if not self.initialized:
                 logger.error("Initialization failed. Cannot add code repository.")
                 return False

        if not self.code_analyzer:
            logger.error("Code Analyzer component is not available. Cannot add repository.")
            return False
        if not self.vector_store: # Code analyzer also needs vector store
            logger.error("Vector Store component is not available. Cannot add repository.")
            return False


        logger.info(f"Attempting to analyze and add code repository: {repo_path} for user: {user_id}")
        try:
            # Analyze repository
            analysis_result = await self.code_analyzer.analyze_repository(
                repo_path=repo_path,
                user_id=user_id # user_id might be used for context or filtering during analysis
            )

            if not analysis_result: # Or check specific attributes of analysis_result
                logger.warning(f"Code repository analysis for '{repo_path}' did not yield results.")
                return False

            # Save to vector store (CodeAnalyzer should handle chunking and embedding of code elements)
            model_id_suffix = self.config.embedding_model.split('/')[-1]
            namespace_to_use = namespace # Use provided or default from signature

            success = await self.code_analyzer.save_to_vector_store(
                analysis=analysis_result,
                vector_store=self.vector_store,
                user_id=user_id,
                model_id=model_id_suffix,
                namespace=namespace_to_use
            )

            if success:
                logger.info(f"✅ Code repository '{Path(repo_path).name}' (Project: {analysis_result.project_name}) analyzed and added to namespace '{namespace_to_use}'.")
            else:
                logger.warning(f"⚠️ Failed to save code repository analysis for '{Path(repo_path).name}' to vector store.")
            return success

        except Exception as e:
            logger.error(f"❌ Failed to analyze/add code repository '{repo_path}': {e}", exc_info=True)
            return False

    async def batch_query(self, queries: List[RAGQuery]) -> List[RAGResponse]:
        """
        Batch Processing für multiple Queries.
        Uses asyncio.gather for parallel execution if config.parallel_processing is True.
        """
        if not self.initialized:
            logger.warning("Orchestrator not initialized. Initializing for batch_query.")
            await self.initialize() # Ensure initialization
            if not self.initialized:
                logger.error("Initialization failed. Cannot process batch query.")
                # Return error responses for all queries
                return [RAGResponse(
                            query=q.query, answer="Error: RAG system not initialized.", sources=[],
                            mode=q.mode, user_id=q.user_id, confidence=0.0, processing_time=0.0,
                            token_count=0, source_count=0, conversation_id=q.conversation_id,
                            debug_info={"error": "System not initialized"}
                        ) for q in queries]


        if not queries:
            return []

        logger.info(f"Starting batch query processing for {len(queries)} queries. Parallel processing: {self.config.parallel_processing}")

        if not self.config.parallel_processing:
            # Sequential processing
            responses: List[RAGResponse] = []
            for rag_q_obj in queries: # Renamed to avoid conflict with self.query
                response = await self.query(rag_q_obj)
                responses.append(response)
            return responses

        # Parallel processing
        tasks = [self.query(rag_q_obj) for rag_q_obj in queries]
        # results can contain RAGResponse objects or Exception objects
        results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

        final_responses: List[RAGResponse] = []
        for i, res_or_exc in enumerate(results_or_exceptions):
            original_query = queries[i]
            if isinstance(res_or_exc, RAGResponse):
                final_responses.append(res_or_exc)
            elif isinstance(res_or_exc, Exception):
                logger.error(f"Error processing query '{original_query.query}' in batch: {res_or_exc}", exc_info=True)
                error_response = RAGResponse(
                    query=original_query.query,
                    answer=f"Error during batch processing: {str(res_or_exc)}",
                    sources=[],
                    mode=original_query.mode,
                    user_id=original_query.user_id,
                    confidence=0.0,
                    processing_time=0.0, # Or measure time until error
                    token_count=0,
                    source_count=0,
                    conversation_id=original_query.conversation_id,
                    debug_info={"error": str(res_or_exc), "traceback": traceback.format_exc() if hasattr(traceback, 'format_exc') else None}
                )
                final_responses.append(error_response)
            else: # Should not happen with return_exceptions=True
                logger.error(f"Unexpected result type in batch processing for query '{original_query.query}': {type(res_or_exc)}")
                # Create a generic error response
                error_response = RAGResponse(
                    query=original_query.query,
                    answer="Unexpected error during batch processing.",
                    sources=[], mode=original_query.mode, user_id=original_query.user_id,
                    confidence=0.0, processing_time=0.0, token_count=0, source_count=0,
                    conversation_id=original_query.conversation_id,
                    debug_info={"error": "Unexpected result type in asyncio.gather"}
                )
                final_responses.append(error_response)

        logger.info(f"Batch query processing completed for {len(queries)} queries.")
        return final_responses

    def _calculate_confidence(self,
                              search_results: List[QueryResult],
                              llm_response: LLMResponse) -> float:
        """
        Berechnet Confidence Score für Response.
        This is a heuristic and can be refined.
        """
        if not search_results and not llm_response.response: # No info at all
            return 0.0
        if not llm_response.response : # LLM failed to generate
            return 0.05 # Minimal confidence if some sources were found but LLM failed

        factors = []

        # Source quality factor (average score of retrieved documents)
        if search_results:
            avg_score = sum(result.score for result in search_results if isinstance(result.score, (int, float))) / len(search_results)
            # Normalize score (assuming scores are typically 0-1, can adjust if different range)
            factors.append(min(max(avg_score, 0.0), 1.0) * 0.4) # Weight 40%
        else: # No sources found
            factors.append(0.0 * 0.4) # Penalty if no sources

        # Number of sources factor (more sources up to a point is better)
        source_count_factor = min(1.0, len(search_results) / 5.0) # Max benefit at 5 sources
        factors.append(source_count_factor * 0.3) # Weight 30%

        # Response length factor (very short responses might be less confident, e.g., "I don't know")
        response_length = len(llm_response.response.split())
        # Penalize very short answers, but don't over-reward long ones.
        if response_length < 5:
            length_factor = 0.2
        elif response_length < 20:
            length_factor = 0.6
        else:
            length_factor = 1.0
        factors.append(length_factor * 0.2) # Weight 20%

        # Base confidence for successful processing and LLM generation
        factors.append(0.1) # Weight 10%

        calculated_confidence = sum(factors)
        return min(max(calculated_confidence, 0.0), 1.0) # Ensure confidence is between 0 and 1

    def _get_cache_key(self, rag_query: RAGQuery) -> str:
        """Generiert Cache Key für Query. Ensures all relevant parts are included."""
        key_parts = [
            rag_query.query,
            rag_query.user_id,
            rag_query.mode.value,
            str(rag_query.namespace or self.config.default_namespace), # Use consistent default
            str(rag_query.k or self.config.default_k),
            json.dumps(rag_query.filters, sort_keys=True) if rag_query.filters else "None", # Consistent serialization
            str(rag_query.conversation_id or "None"),
            str(rag_query.context or "None") # Include explicit context if provided
        ]
        key_string = "|".join(str(part) for part in key_parts)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    def _update_cache(self, cache_key: str, response: RAGResponse) -> None:
        """Updated Response Cache. Implements FIFO eviction if cache is full."""
        if self.response_cache is None: # Cache disabled
            return

        # Don't cache low-confidence or error responses (e.g., confidence < 0.2)
        if response.confidence < 0.2 or response.answer.startswith("Error:"):
            logger.debug(f"Skipping caching for low confidence/error response (key: {cache_key})")
            return

        # Cache size management (FIFO)
        if len(self.response_cache) >= self.max_cache_size:
            # Remove the oldest entry (Python dicts maintain insertion order from 3.7+)
            try:
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]
                logger.debug(f"Cache full. Evicted oldest entry: {oldest_key}")
            except StopIteration: # Should not happen if len >= 1
                pass
        self.response_cache[cache_key] = response
        logger.debug(f"Cached response for key: {cache_key}")

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Holt Statistiken für einen User, primär aus dem Vector Store."""
        if not self.initialized or not self.vector_store:
            logger.warning("Cannot get user stats: Orchestrator or Vector Store not initialized.")
            return {"user_id": user_id, "error": "System or Vector Store not initialized."}

        try:
            # Assuming embedding model ID is needed for vector store stats scoping
            model_id_suffix = self.config.embedding_model.split('/')[-1]
            stats_obj = await self.vector_store.get_store_stats(
                user_id=user_id,
                model_id=model_id_suffix # Or however your vector store scopes stats
            )

            if stats_obj:
                return {
                    "user_id": user_id,
                    "total_vectors": stats_obj.total_vectors,
                    "storage_size_mb": stats_obj.storage_size_mb,
                    "last_updated": stats_obj.last_updated,
                    "query_latency_ms": stats_obj.query_latency_ms, # If provided by store
                    "available_namespaces": await self._get_user_namespaces(user_id, model_id_suffix)
                }
            else:
                return {"user_id": user_id, "message": "No stats returned from vector store.", "available_namespaces": await self._get_user_namespaces(user_id, model_id_suffix)}
        except Exception as e:
            logger.error(f"⚠️ Error getting user stats for '{user_id}': {e}", exc_info=True)
            return {"user_id": user_id, "error": str(e)}

    async def _get_user_namespaces(self, user_id: str, model_id: str) -> List[str]:
        """
        Ermittelt verfügbare Namespaces für User.
        This is a placeholder; actual implementation depends on vector store capabilities.
        """
        # Example: if vector_store has a method like list_namespaces(user_id, model_id)
        # For now, returning common/default ones.
        # if self.vector_store and hasattr(self.vector_store, "list_namespaces_for_user"):
        # try:
        # return await self.vector_store.list_namespaces_for_user(user_id, model_id)
        # except Exception as e:
        # logger.warning(f"Could not dynamically fetch namespaces for user {user_id}: {e}")
        # Fallback to common/default namespaces
        common_namespaces = {self.config.default_namespace, "code_analysis", "research"}
        # You might also scan self.config or other sources for potential namespaces
        return sorted(list(common_namespaces))


    async def clear_user_cache(self, user_id: str) -> bool:
        """Löscht Response Cache Einträge für einen spezifischen User."""
        if self.response_cache is None:
            logger.info("Response cache is disabled. No user cache to clear.")
            return False
        try:
            keys_to_remove = [
                key for key, response_obj in self.response_cache.items() # Renamed to avoid conflict
                if response_obj.user_id == user_id
            ]
            if not keys_to_remove:
                logger.info(f"No cache entries found for user '{user_id}'.")
                return True # Operation successful, nothing to remove

            for key in keys_to_remove:
                del self.response_cache[key]

            logger.info(f"✅ Cleared {len(keys_to_remove)} cache entries for user '{user_id}'.")
            return True
        except Exception as e:
            logger.error(f"❌ Error clearing user cache for '{user_id}': {e}", exc_info=True)
            return False

    async def optimize_performance(self) -> Dict[str, Any]:
        """
        Führt grundlegende Optimierungsaufgaben aus, wie Cache-Bereinigung.
        """
        optimization_results: Dict[str, Any] = {"status": "pending", "timestamp": datetime.now().isoformat()}
        logger.info("🔧 Starting performance optimization tasks...")

        if not self.initialized:
            logger.warning("Orchestrator not initialized. Attempting to initialize for optimization.")
            await self.initialize()
            if not self.initialized:
                logger.error("Initialization failed. Cannot optimize performance.")
                optimization_results["status"] = "failed"
                optimization_results["error"] = "System not initialized for optimization"
                return optimization_results


        # Clear old response cache entries if cache is over 80% full
        if self.response_cache is not None and len(self.response_cache) > self.max_cache_size * 0.8:
            old_size = len(self.response_cache)
            # Keep only most recent entries (approx. 60% of max size)
            # This relies on dicts preserving insertion order (Python 3.7+)
            num_to_keep = int(self.max_cache_size * 0.6)
            if old_size > num_to_keep:
                keys_to_evict = list(self.response_cache.keys())[:old_size - num_to_keep]
                for key in keys_to_evict:
                    del self.response_cache[key]
                optimization_results["response_cache_cleaned"] = old_size - len(self.response_cache)
                logger.info(f"Response cache cleaned: {optimization_results['response_cache_cleaned']} entries removed.")

        # Optimize embedding engine cache (if method exists and cache is large)
        if self.embedding_engine and hasattr(self.embedding_engine, 'get_stats') and hasattr(self.embedding_engine, 'clear_cache'):
            try:
                embedding_stats = self.embedding_engine.get_stats()
                # Example threshold, adjust based on typical cache item size and memory
                if embedding_stats.get("cache_size", 0) > 800: # Assuming cache_size is number of items
                    self.embedding_engine.clear_cache() # Or a more granular clear_old_entries
                    optimization_results["embedding_cache_cleared"] = True
                    logger.info("Embedding engine cache cleared due to size.")
            except Exception as e:
                logger.warning(f"Could not optimize embedding engine cache: {e}", exc_info=True)


        # Optimize LLM handler cache (if method exists and hit rate is low)
        if self.llm_handler and hasattr(self.llm_handler, 'get_performance_stats') and hasattr(self.llm_handler, 'clear_cache'):
            try:
                llm_stats = self.llm_handler.get_performance_stats()
                if llm_stats.get("cache_hit_rate", 1.0) < 0.3 and llm_stats.get("cache_size", 0) > 100: # Low hit rate and substantial size
                    self.llm_handler.clear_cache()
                    optimization_results["llm_cache_cleared"] = True
                    logger.info("LLM handler cache cleared due to low hit rate.")
            except Exception as e:
                logger.warning(f"Could not optimize LLM handler cache: {e}", exc_info=True)

        # Potentially call optimize methods on other components if they exist
        # e.g., if self.vector_store.optimize_indices()

        optimization_results["status"] = "completed"
        optimization_results["timestamp"] = datetime.now().isoformat()
        logger.info("🔧 Performance optimization completed.")
        return optimization_results

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Liefert umfassende Performance-Statistiken des Orchestrators und seiner Komponenten."""
        avg_processing_time = (self.total_processing_time / self.total_queries) if self.total_queries > 0 else 0
        cache_total_lookups = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / cache_total_lookups) if cache_total_lookups > 0 else 0

        stats: Dict[str, Any] = {
            "orchestrator": {
                "total_queries_processed": self.total_queries,
                "total_processing_time_seconds": self.total_processing_time,
                "average_processing_time_seconds": avg_processing_time,
                "response_cache_status": "enabled" if self.config.cache_enabled and self.response_cache is not None else "disabled",
                "response_cache_hits": self.cache_hits,
                "response_cache_misses": self.cache_misses,
                "response_cache_hit_rate": cache_hit_rate,
                "response_cache_size": len(self.response_cache) if self.response_cache is not None else 0,
                "mode_usage": {mode.value: count for mode, count in self.mode_usage.items()},
                "active_conversations_count": len(self.conversations)
            },
            "system_config": {
                "parallel_processing_enabled": self.config.parallel_processing,
                "reranking_enabled": self.config.rerank_enabled,
            },
            "components_overview": {
                "initialized_successfully": self.initialized,
                "status_details": self.component_status.copy() # Shallow copy
            },
            "component_specific_stats": {}
        }

        # Add individual component stats if available and they have a stats method
        component_map = {
            "embedding_engine": self.embedding_engine,
            "vector_store": self.vector_store,
            "llm_handler": self.llm_handler,
            "rerank_engine": self.rerank_engine,
            "document_processor": self.document_processor,
            "code_analyzer": self.code_analyzer,
            "research_assistant": self.research_assistant
        }

        for name, component_instance in component_map.items():
            if component_instance and hasattr(component_instance, 'get_performance_stats'): # Standardized method name
                try:
                    stats["component_specific_stats"][name] = component_instance.get_performance_stats()
                except Exception as e:
                    stats["component_specific_stats"][name] = {"error": f"Failed to get stats: {e}"}
            elif component_instance and hasattr(component_instance, 'get_stats'): # Fallback for embedding_engine
                 try:
                    stats["component_specific_stats"][name] = component_instance.get_stats()
                 except Exception as e:
                    stats["component_specific_stats"][name] = {"error": f"Failed to get stats: {e}"}


        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Umfassender Health Check aller Components, inklusive Testaufrufe."""
        health_status: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy", # Assume healthy initially
            "components": {}
        }
        logger.info("Performing comprehensive health check...")

        if not self.initialized:
            logger.warning("Orchestrator not fully initialized. Health check might be incomplete.")
            # Attempt to initialize if not already, but don't fail health check if it's a quick check
            try:
                if not self.initialized: await self.initialize()
            except Exception as e:
                 logger.error(f"Initialization attempt during health check failed: {e}")
                 health_status["overall_status"] = "unhealthy"
                 health_status["initialization_error"] = str(e)
                 return health_status # Critical failure

        # Test Embedding Engine
        if self.embedding_engine:
            try:
                test_emb_res = await self.embedding_engine.embed(["health check query"])
                if test_emb_res and test_emb_res.embeddings:
                    health_status["components"]["embedding_engine"] = {
                        "status": "healthy", "details": f"Responded with {len(test_emb_res.embeddings[0])}-dim embedding."}
                else:
                    raise ValueError("No embeddings returned.")
            except Exception as e:
                health_status["components"]["embedding_engine"] = {"status": "unhealthy", "error": str(e)}
                health_status["overall_status"] = "degraded"
        else:
            health_status["components"]["embedding_engine"] = {"status": "not_available"}
            if "embedding_engine" not in self.component_status or "Error" in self.component_status.get("embedding_engine", ""):
                 health_status["overall_status"] = "degraded"


        # Test Vector Store
        if self.vector_store:
            try:
                # A light operation, e.g., checking status or a dummy user store
                await self.vector_store.create_user_store(f"health_check_vs_{int(time.time())}", "health_model")
                health_status["components"]["vector_store"] = {"status": "healthy", "details": "Connection and basic operation successful."}
            except Exception as e:
                health_status["components"]["vector_store"] = {"status": "unhealthy", "error": str(e)}
                health_status["overall_status"] = "degraded"
        else:
            health_status["components"]["vector_store"] = {"status": "not_available"}
            if "vector_store" not in self.component_status or "Error" in self.component_status.get("vector_store", ""):
                health_status["overall_status"] = "degraded"

        # Test LLM Handler
        if self.llm_handler:
            try:
                test_req = LLMRequest(prompt="Health check prompt", user_id="health_checker", max_tokens=5)
                test_llm_res = await self.llm_handler.generate_single(test_req)
                if test_llm_res and test_llm_res.response:
                     health_status["components"]["llm_handler"] = {"status": "healthy", "details": f"Responded with {len(test_llm_res.response)} chars."}
                else:
                    raise ValueError("No response from LLM.")
            except Exception as e:
                health_status["components"]["llm_handler"] = {"status": "unhealthy", "error": str(e)}
                health_status["overall_status"] = "degraded"
        else:
            health_status["components"]["llm_handler"] = {"status": "not_available"}
            if "llm_handler" not in self.component_status or "Error" in self.component_status.get("llm_handler", ""):
                health_status["overall_status"] = "degraded"


        # Test Rerank Engine (if enabled and available)
        if self.config.rerank_enabled and self.rerank_engine:
            try:
                test_candidates = [{"id": "doc1", "content": "Test content for rerank.", "score": 0.8}]
                rerank_res = await self.rerank_engine.rerank("health check query", test_candidates, top_k=1)
                if rerank_res and rerank_res.candidates is not None: # Check if candidates attribute exists
                     health_status["components"]["rerank_engine"] = {"status": "healthy", "details": "Rerank successful."}
                else:
                    raise ValueError("Rerank did not return candidates.")
            except Exception as e:
                health_status["components"]["rerank_engine"] = {"status": "unhealthy", "error": str(e)}
                # Don't necessarily mark overall as degraded if reranker fails, as it might be optional
                logger.warning(f"Rerank engine health check failed but system might still be operational: {e}")
        elif self.config.rerank_enabled and not self.rerank_engine:
             health_status["components"]["rerank_engine"] = {"status": "configured_but_not_available"}
        else: # Disabled
            health_status["components"]["rerank_engine"] = {"status": "disabled_by_config"}

        # Check tool components based on their status from initialization
        tool_components_to_check = {
            "document_processor": self.document_processor,
            "code_analyzer": self.code_analyzer,
            "research_assistant": self.research_assistant
        }
        for name, comp_instance in tool_components_to_check.items():
            if comp_instance:
                 health_status["components"][name] = {"status": self.component_status.get(name, "✅ Ready (assumed)")}
                 if "Error" in self.component_status.get(name, ""):
                     health_status["overall_status"] = "degraded"
            else:
                 health_status["components"][name] = {"status": self.component_status.get(name, "not_available")}
                 # If a tool was expected but not available, it might be an issue
                 if "Error" in self.component_status.get(name, ""): # Check if it failed init
                     health_status["overall_status"] = "degraded"


        # Final overall status check
        if any("unhealthy" in comp.get("status", "") for comp in health_status["components"].values()):
            health_status["overall_status"] = "unhealthy"
        elif any("degraded" in comp.get("status", "") for comp in health_status["components"].values()):
             health_status["overall_status"] = "degraded"


        logger.info(f"Health check completed. Overall status: {health_status['overall_status']}")
        return health_status

    async def export_configuration(self, output_path: Union[str, Path]) -> bool:
        """Exportiert aktuelle Konfiguration und Basis-Statistiken in eine JSON-Datei."""
        logger.info(f"Attempting to export configuration to: {output_path}")
        try:
            # Ensure output_path is a Path object for consistency
            output_file_path = Path(output_path)
            output_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

            config_data = {
                "rag_config": asdict(self.config), # Convert RAGConfig dataclass to dict
                "component_initialization_status": self.component_status,
                "current_orchestrator_stats": self.get_comprehensive_stats().get("orchestrator"), # Get orchestrator part
                "export_timestamp": datetime.now().isoformat()
            }

            async with aiofiles.open(output_file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(config_data, indent=4, ensure_ascii=False))

            logger.info(f"✅ Configuration successfully exported to {output_file_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Configuration export failed: {e}", exc_info=True)
            return False

    async def benchmark(self, num_queries_per_mode: int = 2) -> Dict[str, Any]:
        """
        Führt einen einfachen Benchmark für die Haupt-RAG Modi durch.
        Uses a small set of generic queries.
        """
        if not self.initialized:
            logger.error("Cannot run benchmark: Orchestrator not initialized.")
            await self.initialize() # Try to initialize
            if not self.initialized:
                return {"error": "Orchestrator not initialized for benchmark."}

        logger.info(f"🏁 Running RAG Orchestrator benchmark with {num_queries_per_mode} queries per mode...")

        # Generic test queries for different modes
        # In a real benchmark, these would be more diverse and representative
        base_queries = [
            "What is the capital of France and its main attractions?",
            "Explain the concept of photosynthesis in simple terms.",
            "Write a python function to calculate factorial.",
            "Summarize the plot of 'War and Peace'.",
            "What are the latest advancements in renewable energy?"
        ]
        
        benchmark_data: Dict[str, Any] = {"summary": {}, "mode_results": {}}
        total_benchmark_time_start = time.monotonic()
        all_mode_timings = []

        # Test each RAG mode
        for mode in RAGMode:
            if mode == RAGMode.CONVERSATIONAL and num_queries_per_mode > 1: # Need a base query for conversational
                logger.info(f"Skipping extensive conversational benchmark for now, doing 1 query.")
                current_mode_queries_count = 1
            else:
                current_mode_queries_count = num_queries_per_mode

            mode_timings = []
            mode_successes = 0
            logger.info(f"Benchmarking mode: {mode.value}...")

            # Create RAGQuery objects for the mode
            test_rag_queries: List[RAGQuery] = []
            conv_id_for_mode = f"benchmark_conv_{mode.value}_{int(time.time())}"

            for i in range(current_mode_queries_count):
                query_text = base_queries[i % len(base_queries)] # Cycle through base queries
                if mode == RAGMode.CONVERSATIONAL and i > 0: # Follow-up for conversational
                    query_text = f"Tell me more about the previous topic related to {base_queries[(i-1)%len(base_queries)]}."
                
                rag_q = RAGQuery(
                    query=query_text,
                    user_id="benchmark_user",
                    mode=mode,
                    namespace= "benchmark_ns" if mode != RAGMode.CODE_ASSISTANCE else "code_analysis_bm",
                    conversation_id=conv_id_for_mode if mode == RAGMode.CONVERSATIONAL else None,
                    k=3 # Use a small k for benchmark speed
                )
                test_rag_queries.append(rag_q)

            for rag_q_obj in test_rag_queries:
                query_start_time = time.monotonic()
                try:
                    response = await self.query(rag_q_obj)
                    query_time = time.monotonic() - query_start_time
                    mode_timings.append(query_time)
                    all_mode_timings.append(query_time)
                    if response and not response.answer.startswith("Error:"):
                        mode_successes += 1
                except Exception as e:
                    query_time = time.monotonic() - query_start_time
                    logger.error(f"Error during benchmark for query '{rag_q_obj.query}' (mode {mode.value}): {e}", exc_info=True)
                    # Still record timing if possible
                    mode_timings.append(query_time) # Or a penalty time
                    all_mode_timings.append(query_time)


            avg_time_per_query = sum(mode_timings) / len(mode_timings) if mode_timings else 0
            benchmark_data["mode_results"][mode.value] = {
                "queries_run": len(test_rag_queries),
                "successful_queries": mode_successes,
                "total_time_seconds": sum(mode_timings),
                "average_time_per_query_seconds": avg_time_per_query,
                "min_time_seconds": min(mode_timings) if mode_timings else 0,
                "max_time_seconds": max(mode_timings) if mode_timings else 0,
            }
            logger.info(f"Mode {mode.value} benchmark: Avg time {avg_time_per_query:.3f}s over {len(test_rag_queries)} queries.")

        total_benchmark_time = time.monotonic() - total_benchmark_time_start
        benchmark_data["summary"] = {
            "total_queries_run_across_all_modes": len(all_mode_timings),
            "total_benchmark_duration_seconds": total_benchmark_time,
            "overall_average_query_time_seconds": sum(all_mode_timings) / len(all_mode_timings) if all_mode_timings else 0,
            "num_modes_tested": len(RAGMode)
        }
        logger.info(f"🏁 Benchmark finished in {total_benchmark_time:.2f}s. Overall avg query time: {benchmark_data['summary']['overall_average_query_time_seconds']:.3f}s.")
        return benchmark_data

    async def close(self) -> None:
        """Schließt alle Ressourcen und Komponenten ordnungsgemäß."""
        logger.info("🔄 Shutting down RAG Orchestrator and its components...")
        # Close components that might have persistent connections or background tasks
        if self.vector_store and hasattr(self.vector_store, 'close'):
            try:
                await self.vector_store.close()
                logger.info("Vector Store closed.")
            except Exception as e:
                logger.warning(f"Error closing Vector Store: {e}", exc_info=True)

        if self.research_assistant and hasattr(self.research_assistant, 'close'):
            try:
                await self.research_assistant.close() # If it manages browser sessions etc.
                logger.info("Research Assistant closed.")
            except Exception as e:
                logger.warning(f"Error closing Research Assistant: {e}", exc_info=True)
        
        # Other components like embedding_engine, llm_handler might not need explicit close
        # unless they manage external resources directly.

        # Clear in-memory caches and states
        if self.response_cache is not None:
            self.response_cache.clear()
            logger.info("Response cache cleared.")
        self.conversations.clear()
        logger.info("Active conversations cleared.")

        self.initialized = False
        self.component_status = {"orchestrator": "Shutdown"}
        logger.info("✅ RAG Orchestrator shutdown complete.")

# Convenience Builder Pattern (remains largely the same, uses logger now)
class RAGBuilder:
    """Builder Pattern für einfache Konfiguration und Initialisierung des RAG Orchestrators."""
    def __init__(self):
        self._config = RAGConfig() # Use a private attribute for internal config
        logger.debug("RAGBuilder initialized.")

    def with_models(self, embedding_model: str, llm_model: str) -> 'RAGBuilder':
        self._config.embedding_model = embedding_model
        self._config.llm_model = llm_model
        logger.debug(f"Builder: Models set - Embedding='{embedding_model}', LLM='{llm_model}'")
        return self

    def with_vector_store(self, url: str, api_key: Optional[str] = None) -> 'RAGBuilder':
        self._config.vector_store_url = url
        self._config.vector_store_api_key = api_key
        logger.debug(f"Builder: Vector store set - URL='{url}'")
        return self

    def with_performance_settings(self,
                                  batch_size: Optional[int] = None,
                                  max_tokens: Optional[int] = None,
                                  cache_enabled: Optional[bool] = None,
                                  parallel_processing: Optional[bool] = None) -> 'RAGBuilder':
        if batch_size is not None: self._config.batch_size = batch_size
        if max_tokens is not None: self._config.max_tokens = max_tokens
        if cache_enabled is not None: self._config.cache_enabled = cache_enabled
        if parallel_processing is not None: self._config.parallel_processing = parallel_processing
        logger.debug(f"Builder: Performance settings updated - BatchSize={self._config.batch_size}, MaxTokens={self._config.max_tokens}, Cache={self._config.cache_enabled}, Parallel={self._config.parallel_processing}")
        return self

    def with_retrieval_settings(self,
                                default_k: Optional[int] = None,
                                rerank_enabled: Optional[bool] = None,
                                rerank_top_k: Optional[int] = None) -> 'RAGBuilder':
        if default_k is not None: self._config.default_k = default_k
        if rerank_enabled is not None: self._config.rerank_enabled = rerank_enabled
        if rerank_top_k is not None: self._config.rerank_top_k = rerank_top_k
        logger.debug(f"Builder: Retrieval settings updated - DefaultK={self._config.default_k}, Rerank={self._config.rerank_enabled}, RerankTopK={self._config.rerank_top_k}")
        return self
    
    def with_custom_config(self, custom_config: RAGConfig) -> 'RAGBuilder':
        """Allows setting a full RAGConfig object directly."""
        self._config = custom_config
        logger.debug("Builder: Custom RAGConfig object applied.")
        return self

    async def build(self) -> MLXRAGOrchestrator:
        """Baut und initialisiert den MLXRAGOrchestrator mit der aktuellen Konfiguration."""
        logger.info("Building RAG Orchestrator from RAGBuilder...")
        orchestrator = MLXRAGOrchestrator(self._config)
        try:
            await orchestrator.initialize()
            logger.info("RAG Orchestrator built and initialized successfully via RAGBuilder.")
        except Exception as e:
            logger.error(f"Failed to build or initialize RAG Orchestrator via RAGBuilder: {e}", exc_info=True)
            # Depending on desired behavior, either raise e or return a non-initialized orchestrator
            # For robustness, return the orchestrator instance even if init fails, its `initialized` flag will be False.
        return orchestrator

# Example Usage (updated to use logger)
async def example_usage():
    """Beispiele für RAG Orchestrator Usage"""
    logger.info("--- Starting RAG Orchestrator Example Usage ---")

    # Method 1: Direct initialization
    config = RAGConfig(
        embedding_model="mlx-community/gte-small", # Ensure this model is accessible
        llm_model="mlx-community/gemma-2-9b-it-4bit", # Ensure this model is accessible
        vector_store_url="http://localhost:8000", # Ensure your vector store is running here
        rerank_enabled=True,
        cache_enabled=True,
        default_namespace="example_docs"
    )
    orchestrator = MLXRAGOrchestrator(config)

    # Create dummy files for testing add_document and add_code_repository
    Path("example_docs").mkdir(exist_ok=True)
    Path("example_repo").mkdir(exist_ok=True)
    with open("example_docs/example.pdf", "w") as f: f.write("This is a test PDF content for MLX RAG.")
    with open("example_docs/another.txt", "w") as f: f.write("Another text file with important keywords.")
    with open("example_repo/main.py", "w") as f: f.write("def hello_world():\n    print('Hello from MLX RAG example repo!')")


    try:
        # Initialize explicitly or it will be done on first query/add
        await orchestrator.initialize()
        if not orchestrator.initialized:
            logger.error("Orchestrator failed to initialize in example. Exiting.")
            return

        # Add documents
        # Note: Ensure 'example.pdf' exists or use a valid path
        pdf_added = await orchestrator.add_document(
            file_path="example_docs/example.pdf", # Make sure this file exists
            user_id="user_123",
            namespace="research_papers", # Specific namespace for this document
            metadata={"category": "research", "priority": "high", "year": 2024}
        )
        logger.info(f"Document 'example.pdf' added status: {pdf_added}")

        txt_added = await orchestrator.add_document(
            file_path="example_docs/another.txt",
            user_id="user_123",
            namespace="general_notes",
            metadata={"topic": "keywords"}
        )
        logger.info(f"Document 'another.txt' added status: {txt_added}")


        # Add code repository (ensure 'path/to/repository' is valid or a dummy one)
        repo_added = await orchestrator.add_code_repository(
            repo_path="example_repo", # Make sure this path exists and has some files
            user_id="user_123",
            namespace="project_alpha_code" # Specific namespace for this repo
        )
        logger.info(f"Code repository 'example_repo' added status: {repo_added}")

        # --- Define Queries ---
        queries_to_test = [
            RAGQuery(
                query="What is MLX RAG?",
                user_id="user_123",
                mode=RAGMode.DOCUMENT_QA,
                namespace="research_papers" # Query within the specific namespace
            ),
            RAGQuery(
                query="Find important keywords.",
                user_id="user_123",
                mode=RAGMode.DOCUMENT_QA,
                namespace="general_notes"
            ),
            RAGQuery(
                query="Explain the hello_world function.",
                user_id="user_123",
                mode=RAGMode.CODE_ASSISTANCE,
                namespace="project_alpha_code" # Query within the code's namespace
            ),
            RAGQuery(
                query="What are the latest trends in local AI models?",
                user_id="user_123",
                mode=RAGMode.RESEARCH # This uses the ResearchAssistant
            ),
            RAGQuery( # First message in a conversation
                query="Tell me about unified memory on Apple Silicon.",
                user_id="user_456", # Different user
                mode=RAGMode.CONVERSATIONAL,
                conversation_id="conv_apple_silicon" # Start a new conversation
            ),
            RAGQuery( # Follow-up
                query="How does it benefit MLX?",
                user_id="user_456",
                mode=RAGMode.CONVERSATIONAL,
                conversation_id="conv_apple_silicon" # Continue existing conversation
            ),
            RAGQuery(
                query="Compare Python web frameworks and show a simple Flask example.",
                user_id="user_789",
                mode=RAGMode.HYBRID # Hybrid should try to combine doc search, code search, maybe research
            )
        ]

        # Process single queries
        logger.info("\n--- Processing Single Queries ---")
        for rag_q in queries_to_test:
            logger.info(f"\nSubmitting Query: '{rag_q.query}' (Mode: {rag_q.mode.value}, Namespace: {rag_q.namespace or 'default'})")
            response = await orchestrator.query(rag_q)
            logger.info(f"Answer: {response.answer[:300]}...") # Log a snippet
            logger.info(f"Sources Found: {response.source_count}, Confidence: {response.confidence:.2f}, Time: {response.processing_time:.3f}s, Cache Hit: {response.cache_hit}")
            if response.sources:
                logger.debug(f"First source example: ID='{response.sources[0].get('id')}', Title='{response.sources[0].get('title')}'")

        # Batch processing
        logger.info("\n--- Processing Batch Queries ---")
        # Re-use some queries for batch, ensure they are distinct enough if cache is on
        batch_queries_list = [
            RAGQuery(query="What is MLX?", user_id="batch_user", mode=RAGMode.DOCUMENT_QA, namespace="research_papers"),
            RAGQuery(query="How to optimize Python code?", user_id="batch_user", mode=RAGMode.CODE_ASSISTANCE, namespace="project_alpha_code"),
            RAGQuery(query="Latest AI news", user_id="batch_user", mode=RAGMode.RESEARCH)
        ]
        batch_responses = await orchestrator.batch_query(batch_queries_list)
        logger.info(f"Batch processed {len(batch_responses)} queries.")
        for i, resp in enumerate(batch_responses):
            logger.info(f"Batch Query {i+1} ('{resp.query}'): Answer snippet: {resp.answer[:100]}..., Time: {resp.processing_time:.3f}s")


        # Health check
        logger.info("\n--- Performing Health Check ---")
        health = await orchestrator.health_check()
        logger.info(f"Health Status: {health.get('overall_status', 'Unknown')}")
        logger.debug(f"Detailed Health: {json.dumps(health, indent=2)}")

        # Performance stats
        logger.info("\n--- Retrieving Comprehensive Stats ---")
        stats = orchestrator.get_comprehensive_stats()
        logger.info(f"Total Queries Processed by Orchestrator: {stats.get('orchestrator', {}).get('total_queries_processed', 0)}")
        logger.info(f"Response Cache Hit Rate: {stats.get('orchestrator', {}).get('response_cache_hit_rate', 0):.2%}")
        logger.debug(f"Detailed Stats: {json.dumps(stats, indent=2)}")

        # Benchmark
        logger.info("\n--- Running Benchmark ---")
        # Reduce num_queries_per_mode for quicker example run
        benchmark_results = await orchestrator.benchmark(num_queries_per_mode=1)
        logger.info(f"Benchmark Summary: {benchmark_results.get('summary', {})}")
        logger.debug(f"Detailed Benchmark Results: {json.dumps(benchmark_results, indent=2)}")


        # Export configuration
        logger.info("\n--- Exporting Configuration ---")
        export_success = await orchestrator.export_configuration("rag_orchestrator_config_export.json")
        logger.info(f"Configuration export successful: {export_success}")

    finally:
        logger.info("--- Closing RAG Orchestrator in Example ---")
        if 'orchestrator' in locals() and orchestrator.initialized: # Check if orchestrator was defined and initialized
            await orchestrator.close()
        # Clean up dummy files
        Path("example_docs/example.pdf").unlink(missing_ok=True)
        Path("example_docs/another.txt").unlink(missing_ok=True)
        Path("example_repo/main.py").unlink(missing_ok=True)
        Path("example_docs").rmdir() # Fails if not empty, which is fine
        Path("example_repo").rmdir() # Fails if not empty

async def builder_example():
    """Beispiel mit Builder Pattern"""
    logger.info("--- Starting RAG Orchestrator Builder Example ---")
    orchestrator_via_builder: Optional[MLXRAGOrchestrator] = None
    try:
        orchestrator_via_builder = await (RAGBuilder()
                              .with_models("mlx-community/gte-large", "mlx-community/gemma-2-27b-it-4bit")
                              .with_vector_store("http://localhost:9000", "optional-api-key") # Different port for example
                              .with_performance_settings(batch_size=10, cache_enabled=False, parallel_processing=False)
                              .with_retrieval_settings(default_k=3, rerank_enabled=False)
                              .build())

        if not orchestrator_via_builder or not orchestrator_via_builder.initialized:
            logger.error("Failed to build or initialize orchestrator via builder. Exiting builder example.")
            return

        # Use orchestrator
        query_obj = RAGQuery(
            query="Explain the concept of quantum entanglement in simple terms.",
            user_id="user_builder_example",
            mode=RAGMode.HYBRID # Hybrid will try its best with available components
        )
        response = await orchestrator_via_builder.query(query_obj)
        logger.info(f"Response from Builder Orchestrator: {response.answer[:300]}...")
        logger.info(f"Confidence: {response.confidence:.2f}, Time: {response.processing_time:.3f}s")

    except Exception as e:
        logger.error(f"Error in builder example: {e}", exc_info=True)
    finally:
        logger.info("--- Closing RAG Orchestrator in Builder Example ---")
        if orchestrator_via_builder and orchestrator_via_builder.initialized:
            await orchestrator_via_builder.close()

if __name__ == "__main__":
    # To run examples, you would need actual MLX components or ensure mocks are sufficient.
    # The mock components are very basic.
    import traceback # For RAGResponse debug_info

    # Create a dummy file for add_document to work in the example
    # Path("example.pdf").touch() # Done within example_usage now

    logger.info("Starting main execution for RAG Orchestrator examples.")
    # asyncio.run(example_usage())
    # asyncio.run(builder_example())
    logger.info("Example runs commented out. Uncomment to run with appropriate setup.")
    logger.info("If running, ensure vector store (e.g., Qdrant, Weaviate) is accessible if not using full mocks.")
    logger.info("Also ensure MLX models are downloaded or paths are correct if not using full mocks.")

