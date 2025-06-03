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

# MLX Components
from mlx_components.embedding_engine import MLXEmbeddingEngine, EmbeddingConfig, EmbeddingResult
from mlx_components.vector_store import MLXVectorStore, VectorStoreConfig, QueryResult
from mlx_components.llm_handler import MLXLLMHandler, LLMConfig, LLMRequest, LLMResponse
from mlx_components.rerank_engine import MLXRerankEngine, ReRankConfig, RerankResult

# Tools
from tools.document_processor import MLXDocumentProcessor, ProcessingConfig, ProcessedDocument
from tools.code_analyzer import MLXCodeAnalyzer, CodeAnalysisConfig, RepositoryAnalysis
from tools.research_assistant import MLXResearchAssistant, ResearchConfig, ResearchResult

class RAGMode(Enum):
    """RAG Operation Modi"""
    DOCUMENT_QA = "document_qa"           # Standard Document Q&A
    CODE_ASSISTANCE = "code_assistance"   # Code-spezifische Hilfe
    RESEARCH = "research"                 # Web Research mit RAG
    CONVERSATIONAL = "conversational"    # Chat mit Kontext
    HYBRID = "hybrid"                     # Kombiniert mehrere Modi

@dataclass
class RAGConfig:
    """Zentrale Konfiguration für RAG Orchestrator"""
    # Model Configurations
    embedding_model: str = "mlx-community/gte-small"
    llm_model: str = "mlx-community/gemma-2-9b-it-4bit"
    
    # Vector Store
    vector_store_url: str = "http://localhost:8000"
    vector_store_api_key: Optional[str] = None
    
    # Processing Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_length: int = 8000
    
    # Retrieval Settings
    default_k: int = 5
    max_k: int = 20
    rerank_enabled: bool = True
    rerank_top_k: int = 10
    
    # Generation Settings
    max_tokens: int = 1024
    temperature: float = 0.1
    batch_size: int = 5
    
    # Performance Settings
    cache_enabled: bool = True
    parallel_processing: bool = True
    
    # User Management
    enable_user_isolation: bool = True
    default_namespace: str = "default"

@dataclass
class RAGQuery:
    """RAG Query Definition"""
    query: str
    user_id: str
    mode: RAGMode = RAGMode.DOCUMENT_QA
    namespace: Optional[str] = None
    context: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    k: Optional[int] = None
    include_sources: bool = True
    conversation_id: Optional[str] = None
    follow_up: bool = False

@dataclass
class RAGResponse:
    """RAG Response mit allen Metadaten"""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    mode: RAGMode
    user_id: str
    confidence: float
    processing_time: float
    token_count: int
    source_count: int
    cache_hit: bool = False
    conversation_id: Optional[str] = None
    follow_up_questions: List[str] = None
    related_topics: List[str] = None
    debug_info: Optional[Dict[str, Any]] = None

class MLXRAGOrchestrator:
    """
    High-Performance RAG Orchestrator für MLX Ecosystem
    
    Features:
    - Unified Interface für alle RAG Operations
    - Multi-Mode Support (Documents, Code, Research, Chat)
    - Intelligent Component Coordination
    - Performance Optimization & Caching
    - User Isolation & Multi-Tenancy
    - Comprehensive Monitoring
    - Brain System Integration Ready
    """
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        
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
        self.component_status = {}
        
        # Performance Metrics
        self.total_queries = 0
        self.total_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.mode_usage = {mode: 0 for mode in RAGMode}
        
        # Response Cache
        self.response_cache = {} if self.config.cache_enabled else None
        self.max_cache_size = 1000
        
        # Active conversations
        self.conversations = {}
    
    async def initialize(self) -> None:
        """
        Initialisiert alle Components mit optimaler Konfiguration
        """
        if self.initialized:
            return
        
        print("🚀 Initializing MLX RAG Orchestrator...")
        start_time = time.time()
        
        try:
            # Initialize Core Components
            await self._initialize_core_components()
            
            # Initialize Tool Components
            await self._initialize_tool_components()
            
            # Verify Component Health
            await self._verify_component_health()
            
            self.initialized = True
            init_time = time.time() - start_time
            
            print(f"✅ RAG Orchestrator initialized in {init_time:.2f}s")
            self._print_component_status()
            
        except Exception as e:
            print(f"❌ Failed to initialize RAG Orchestrator: {e}")
            raise
    
    async def _initialize_core_components(self) -> None:
        """Initialisiert Core MLX Components"""
        
        # Embedding Engine
        embedding_config = EmbeddingConfig(
            model_path=self.config.embedding_model,
            batch_size=32,
            cache_embeddings=self.config.cache_enabled,
            normalize_embeddings=True
        )
        self.embedding_engine = MLXEmbeddingEngine(embedding_config)
        await self.embedding_engine.initialize()
        self.component_status["embedding_engine"] = "✅ Ready"
        
        # Vector Store
        vector_config = VectorStoreConfig(
            base_url=self.config.vector_store_url,
            api_key=self.config.vector_store_api_key,
            batch_size=100,
            default_k=self.config.default_k
        )
        self.vector_store = MLXVectorStore(vector_config)
        await self.vector_store.initialize()
        self.component_status["vector_store"] = "✅ Ready"
        
        # LLM Handler
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
        
        # Re-ranking Engine
        if self.config.rerank_enabled:
            rerank_config = ReRankConfig(
                top_k=self.config.rerank_top_k,
                diversity_factor=0.3,
                enable_diversity_rerank=True
            )
            self.rerank_engine = MLXRerankEngine(rerank_config)
            await self.rerank_engine.initialize()
            self.component_status["rerank_engine"] = "✅ Ready"
        else:
            self.component_status["rerank_engine"] = "⚠️ Disabled"
    
    async def _initialize_tool_components(self) -> None:
        """Initialisiert Tool Components"""
        
        # Document Processor
        doc_config = ProcessingConfig(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            preserve_structure=True,
            auto_summarize=True,
            embedding_model=self.config.embedding_model
        )
        self.document_processor = MLXDocumentProcessor(doc_config, self.embedding_engine)
        self.component_status["document_processor"] = "✅ Ready"
        
        # Code Analyzer
        code_config = CodeAnalysisConfig(
            generate_embeddings=True,
            embedding_model=self.config.embedding_model,
            analyze_complexity=True
        )
        self.code_analyzer = MLXCodeAnalyzer(code_config, self.embedding_engine, self.document_processor)
        self.component_status["code_analyzer"] = "✅ Ready"
        
        # Research Assistant
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
            self.rerank_engine, 
            self.document_processor
        )
        self.component_status["research_assistant"] = "✅ Ready"
    
    async def _verify_component_health(self) -> None:
        """Überprüft Health aller Components"""
        
        # Test embedding
        try:
            test_embedding = await self.embedding_engine.embed(["test"])
            if test_embedding.embeddings:
                self.component_status["embedding_engine"] += " (Tested)"
        except Exception as e:
            self.component_status["embedding_engine"] = f"❌ Error: {e}"
        
        # Test vector store
        try:
            await self.vector_store.create_user_store("health_check", "test_model")
            self.component_status["vector_store"] += " (Connected)"
        except Exception as e:
            self.component_status["vector_store"] = f"⚠️ Connection issue: {e}"
        
        # Test LLM
        try:
            test_request = LLMRequest(
                prompt="Hello",
                user_id="health_check",
                max_tokens=10
            )
            test_response = await self.llm_handler.generate_single(test_request)
            if test_response.response:
                self.component_status["llm_handler"] += " (Tested)"
        except Exception as e:
            self.component_status["llm_handler"] = f"❌ Error: {e}"
    
    def _print_component_status(self) -> None:
        """Gibt Component Status aus"""
        print("\n📊 Component Status:")
        for component, status in self.component_status.items():
            print(f"   {component}: {status}")
        print()
    
    async def query(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Hauptfunktion: Verarbeitet RAG Query mit automatischer Mode-Detection
        """
        start_time = time.time()
        
        if not self.initialized:
            await self.initialize()
        
        # Check cache first
        cache_key = self._get_cache_key(rag_query)
        if self.response_cache and cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            cached_response.cache_hit = True
            self.cache_hits += 1
            return cached_response
        
        try:
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
                raise ValueError(f"Unsupported RAG mode: {rag_query.mode}")
            
            # Calculate processing time
            response.processing_time = time.time() - start_time
            
            # Update metrics
            self.total_queries += 1
            self.total_processing_time += response.processing_time
            self.mode_usage[rag_query.mode] += 1
            self.cache_misses += 1
            
            # Cache response
            if self.response_cache:
                self._update_cache(cache_key, response)
            
            print(f"✅ RAG Query processed in {response.processing_time:.2f}s (mode: {rag_query.mode.value})")
            
            return response
            
        except Exception as e:
            print(f"❌ RAG Query failed: {e}")
            
            # Return error response
            return RAGResponse(
                query=rag_query.query,
                answer=f"Error processing query: {str(e)}",
                sources=[],
                mode=rag_query.mode,
                user_id=rag_query.user_id,
                confidence=0.0,
                processing_time=time.time() - start_time,
                token_count=0,
                source_count=0,
                conversation_id=rag_query.conversation_id
            )
    
    async def _handle_document_qa(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Standard Document Q&A Pipeline
        """
        # 1. Generate query embedding
        query_embedding = await self.embedding_engine.embed([rag_query.query])
        
        # 2. Vector search
        k = rag_query.k or self.config.default_k
        namespace = rag_query.namespace or self.config.default_namespace
        
        search_results = await self.vector_store.query(
            user_id=rag_query.user_id,
            model_id=self.config.embedding_model.split('/')[-1],
            query_vector=query_embedding.embeddings[0],
            k=min(k * 2, self.config.max_k),  # Get more for reranking
            filters=rag_query.filters,
            namespace=namespace
        )
        
        # 3. Re-ranking (if enabled)
        if self.rerank_engine and len(search_results) > 1:
            candidates = [
                {
                    "id": result.id,
                    "content": result.metadata.get("text", ""),
                    "metadata": result.metadata,
                    "score": result.score
                } for result in search_results
            ]
            
            rerank_result = await self.rerank_engine.rerank(
                query=rag_query.query,
                candidates=candidates,
                top_k=k
            )
            
            # Map back to search results
            reranked_results = []
            for candidate in rerank_result.candidates:
                for result in search_results:
                    if result.id == candidate.id:
                        reranked_results.append(result)
                        break
            
            final_results = reranked_results
        else:
            final_results = search_results[:k]
        
        # 4. Build context
        context_docs = []
        for result in final_results:
            context_docs.append({
                "content": result.metadata.get("text", ""),
                "metadata": result.metadata
            })
        
        # 5. Generate response
        llm_response = await self.llm_handler.rag_generate(
            query=rag_query.query,
            context_documents=context_docs,
            user_id=rag_query.user_id,
            system_prompt="You are a helpful AI assistant. Answer questions based on the provided context."
        )
        
        # 6. Prepare sources
        sources = []
        for result in final_results:
            source = {
                "id": result.id,
                "title": result.metadata.get("title", "Unknown"),
                "content": result.metadata.get("text", "")[:200] + "...",
                "metadata": result.metadata,
                "score": result.score
            }
            sources.append(source)
        
        return RAGResponse(
            query=rag_query.query,
            answer=llm_response.response,
            sources=sources,
            mode=rag_query.mode,
            user_id=rag_query.user_id,
            confidence=self._calculate_confidence(final_results, llm_response),
            processing_time=0.0,  # Will be set by caller
            token_count=llm_response.token_count,
            source_count=len(final_results),
            conversation_id=rag_query.conversation_id
        )
    
    async def _handle_code_assistance(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Code-spezifische Assistance Pipeline
        """
        # Similar to document_qa but with code-specific context and prompts
        query_embedding = await self.embedding_engine.embed([rag_query.query])
        
        # Search in code-specific namespace
        namespace = rag_query.namespace or "code_analysis"
        
        search_results = await self.vector_store.query(
            user_id=rag_query.user_id,
            model_id=self.config.embedding_model.split('/')[-1],
            query_vector=query_embedding.embeddings[0],
            k=self.config.default_k,
            filters={
                **rag_query.filters or {},
                "type": "code_element"  # Filter for code elements
            },
            namespace=namespace
        )
        
        # Build code context
        context_docs = []
        for result in search_results:
            context_docs.append({
                "content": result.metadata.get("text", ""),
                "metadata": result.metadata
            })
        
        # Code-specific system prompt
        system_prompt = """You are an expert code assistant. Help with coding questions using the provided code context.
        - Explain code functionality clearly
        - Suggest improvements when appropriate
        - Provide working code examples
        - Reference specific functions/classes from the context"""
        
        llm_response = await self.llm_handler.rag_generate(
            query=rag_query.query,
            context_documents=context_docs,
            user_id=rag_query.user_id,
            system_prompt=system_prompt
        )
        
        sources = [
            {
                "id": result.id,
                "title": result.metadata.get("element_name", "Code Element"),
                "file_path": result.metadata.get("file_path", "Unknown"),
                "language": result.metadata.get("language", "Unknown"),
                "content": result.metadata.get("text", "")[:300] + "...",
                "metadata": result.metadata,
                "score": result.score
            } for result in search_results
        ]
        
        return RAGResponse(
            query=rag_query.query,
            answer=llm_response.response,
            sources=sources,
            mode=rag_query.mode,
            user_id=rag_query.user_id,
            confidence=self._calculate_confidence(search_results, llm_response),
            processing_time=0.0,
            token_count=llm_response.token_count,
            source_count=len(search_results),
            conversation_id=rag_query.conversation_id
        )
    
    async def _handle_research(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Web Research Pipeline mit RAG
        """
        from tools.research_assistant import SearchQuery
        
        # Create search query
        search_query = SearchQuery(
            query=rag_query.query,
            user_id=rag_query.user_id,
            max_results=10,
            context=rag_query.context
        )
        
        # Perform research
        research_result = await self.research_assistant.research(search_query)
        
        # Convert to RAG response format
        sources = [
            {
                "title": source["title"],
                "url": source["url"],
                "domain": source["domain"],
                "type": "web_research"
            } for source in research_result.sources_used
        ]
        
        return RAGResponse(
            query=rag_query.query,
            answer=research_result.synthesized_answer,
            sources=sources,
            mode=rag_query.mode,
            user_id=rag_query.user_id,
            confidence=research_result.confidence_score,
            processing_time=0.0,
            token_count=len(research_result.synthesized_answer.split()),
            source_count=len(research_result.sources_used),
            follow_up_questions=research_result.follow_up_questions,
            related_topics=research_result.related_topics,
            conversation_id=rag_query.conversation_id
        )
    
    async def _handle_conversational(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Conversational RAG mit Chat History
        """
        # Get conversation history
        conversation_id = rag_query.conversation_id or f"conv_{rag_query.user_id}_{int(time.time())}"
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        conversation_history = self.conversations[conversation_id]
        
        # Build conversational context
        context_parts = []
        
        # Add recent conversation turns
        for turn in conversation_history[-3:]:  # Last 3 turns
            context_parts.append(f"Human: {turn['query']}")
            context_parts.append(f"Assistant: {turn['response']}")
        
        # Add current query
        context_parts.append(f"Human: {rag_query.query}")
        
        conversation_context = "\n".join(context_parts)
        
        # Vector search with conversational context
        search_text = f"{conversation_context}\n\nCurrent question: {rag_query.query}"
        query_embedding = await self.embedding_engine.embed([search_text])
        
        search_results = await self.vector_store.query(
            user_id=rag_query.user_id,
            model_id=self.config.embedding_model.split('/')[-1],
            query_vector=query_embedding.embeddings[0],
            k=self.config.default_k,
            filters=rag_query.filters,
            namespace=rag_query.namespace or self.config.default_namespace
        )
        
        # Build context
        context_docs = []
        for result in search_results:
            context_docs.append({
                "content": result.metadata.get("text", ""),
                "metadata": result.metadata
            })
        
        # Conversational system prompt
        system_prompt = f"""You are a helpful AI assistant in a conversation. Use the conversation history and provided context to give relevant, contextual responses.
        
        Conversation so far:
        {conversation_context}
        
        Provide a natural, conversational response that acknowledges the conversation flow."""
        
        llm_response = await self.llm_handler.rag_generate(
            query=rag_query.query,
            context_documents=context_docs,
            user_id=rag_query.user_id,
            system_prompt=system_prompt
        )
        
        # Update conversation history
        self.conversations[conversation_id].append({
            "query": rag_query.query,
            "response": llm_response.response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Limit conversation history
        if len(self.conversations[conversation_id]) > 10:
            self.conversations[conversation_id] = self.conversations[conversation_id][-10:]
        
        sources = [
            {
                "id": result.id,
                "title": result.metadata.get("title", "Context"),
                "content": result.metadata.get("text", "")[:200] + "...",
                "metadata": result.metadata,
                "score": result.score
            } for result in search_results
        ]
        
        return RAGResponse(
            query=rag_query.query,
            answer=llm_response.response,
            sources=sources,
            mode=rag_query.mode,
            user_id=rag_query.user_id,
            confidence=self._calculate_confidence(search_results, llm_response),
            processing_time=0.0,
            token_count=llm_response.token_count,
            source_count=len(search_results),
            conversation_id=conversation_id
        )
    
    async def _handle_hybrid(self, rag_query: RAGQuery) -> RAGResponse:
        """
        Hybrid Pipeline - kombiniert multiple Approaches
        """
        # Try multiple approaches and combine results
        tasks = []
        
        # Document search
        doc_query = RAGQuery(
            query=rag_query.query,
            user_id=rag_query.user_id,
            mode=RAGMode.DOCUMENT_QA,
            namespace=rag_query.namespace,
            filters=rag_query.filters,
            k=3  # Fewer from each source
        )
        tasks.append(("document", self._handle_document_qa(doc_query)))
        
        # Code search (if relevant)
        if any(keyword in rag_query.query.lower() for keyword in ["code", "function", "class", "programming", "implementation"]):
            code_query = RAGQuery(
                query=rag_query.query,
                user_id=rag_query.user_id,
                mode=RAGMode.CODE_ASSISTANCE,
                namespace="code_analysis",
                k=2
            )
            tasks.append(("code", self._handle_code_assistance(code_query)))
        
        # Execute searches in parallel
        results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
        
        # Combine results
        all_sources = []
        context_parts = []
        
        for i, (source_type, _) in enumerate(tasks):
            if not isinstance(results[i], Exception):
                response = results[i]
                
                # Add sources with type annotation
                for source in response.sources:
                    source["source_type"] = source_type
                    all_sources.append(source)
                
                # Add to context
                if response.sources:
                    context_parts.append(f"From {source_type} sources: {response.answer[:300]}...")
        
        # Generate combined response
        if context_parts:
            combined_context = "\n\n".join(context_parts)
            
            synthesis_prompt = f"""
            Based on information from multiple sources, provide a comprehensive answer to the user's question.
            
            Question: {rag_query.query}
            
            Information gathered:
            {combined_context}
            
            Provide a well-structured answer that synthesizes the information from all sources.
            """
            
            synthesis_request = LLMRequest(
                prompt=synthesis_prompt,
                user_id=rag_query.user_id,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            final_response = await self.llm_handler.generate_single(synthesis_request)
            answer = final_response.response
            token_count = final_response.token_count
        else:
            answer = "No relevant information found across available sources."
            token_count = 0
        
        # Calculate combined confidence
        confidence = 0.0
        if all_sources:
            confidence = sum(source.get("score", 0.5) for source in all_sources) / len(all_sources)
        
        return RAGResponse(
            query=rag_query.query,
            answer=answer,
            sources=all_sources,
            mode=rag_query.mode,
            user_id=rag_query.user_id,
            confidence=confidence,
            processing_time=0.0,
            token_count=token_count,
            source_count=len(all_sources),
            conversation_id=rag_query.conversation_id
        )
    
    async def add_document(self, 
                         file_path: Union[str, Path], 
                         user_id: str,
                         namespace: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Fügt Dokument zum RAG System hinzu
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Process document
            processed_doc = await self.document_processor.process_document(
                file_path=file_path,
            async def add_code_repository(self, 
                                repo_path: Union[str, Path], 
                                user_id: str) -> bool:
        """
        Analysiert und fügt Code Repository hinzu
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Analyze repository
            analysis = await self.code_analyzer.analyze_repository(
                repo_path=repo_path,
                user_id=user_id
            )
            
            # Save to vector store
            success = await self.code_analyzer.save_to_vector_store(
                analysis=analysis,
                vector_store=self.vector_store,
                user_id=user_id,
                model_id=self.config.embedding_model.split('/')[-1]
            )
            
            if success:
                print(f"✅ Code repository analyzed and added: {analysis.project_name}")
            
            return success
            
        except Exception as e:
            print(f"❌ Failed to analyze repository: {e}")
            return False
    
    async def batch_query(self, queries: List[RAGQuery]) -> List[RAGResponse]:
        """
        Batch Processing für multiple Queries
        """
        if not self.initialized:
            await self.initialize()
        
        if not self.config.parallel_processing:
            # Sequential processing
            responses = []
            for query in queries:
                response = await self.query(query)
                responses.append(response)
            return responses
        
        # Parallel processing
        tasks = [self.query(query) for query in queries]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                error_response = RAGResponse(
                    query=queries[i].query,
                    answer=f"Error: {str(response)}",
                    sources=[],
                    mode=queries[i].mode,
                    user_id=queries[i].user_id,
                    confidence=0.0,
                    processing_time=0.0,
                    token_count=0,
                    source_count=0
                )
                final_responses.append(error_response)
            else:
                final_responses.append(response)
        
        return final_responses
    
    def _calculate_confidence(self, 
                            search_results: List[QueryResult], 
                            llm_response: LLMResponse) -> float:
        """
        Berechnet Confidence Score für Response
        """
        factors = []
        
        # Source quality factor
        if search_results:
            avg_score = sum(result.score for result in search_results) / len(search_results)
            factors.append(avg_score * 0.4)
        
        # Number of sources factor
        source_factor = min(1.0, len(search_results) / 5.0)
        factors.append(source_factor * 0.3)
        
        # Response length factor (reasonable length indicates confidence)
        response_length = len(llm_response.response.split())
        length_factor = min(1.0, response_length / 100.0)
        factors.append(length_factor * 0.2)
        
        # Processing success factor
        factors.append(0.1)  # Base confidence for successful processing
        
        return sum(factors)
    
    def _get_cache_key(self, rag_query: RAGQuery) -> str:
        """
        Generiert Cache Key für Query
        """
        key_parts = [
            rag_query.query,
            rag_query.user_id,
            rag_query.mode.value,
            str(rag_query.namespace),
            str(rag_query.k),
            str(rag_query.filters)
        ]
        
        key_string = "|".join(str(part) for part in key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_cache(self, cache_key: str, response: RAGResponse) -> None:
        """
        Updated Response Cache
        """
        if not self.response_cache:
            return
        
        # Cache size management
        if len(self.response_cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.response_cache.keys())[:len(self.response_cache) - self.max_cache_size + 1]
            for key in oldest_keys:
                del self.response_cache[key]
        
        # Don't cache error responses
        if response.confidence > 0.1:
            self.response_cache[cache_key] = response
    
    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Holt Statistiken für einen User
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Get vector store stats
            stats = await self.vector_store.get_store_stats(
                user_id=user_id,
                model_id=self.config.embedding_model.split('/')[-1]
            )
            
            return {
                "user_id": user_id,
                "total_vectors": stats.total_vectors if stats else 0,
                "storage_size_mb": stats.storage_size_mb if stats else 0,
                "last_updated": stats.last_updated if stats else None,
                "query_latency_ms": stats.query_latency_ms if stats else 0,
                "available_namespaces": await self._get_user_namespaces(user_id)
            }
            
        except Exception as e:
            print(f"⚠️ Error getting user stats: {e}")
            return {"user_id": user_id, "error": str(e)}
    
    async def _get_user_namespaces(self, user_id: str) -> List[str]:
        """
        Ermittelt verfügbare Namespaces für User
        """
        # This would need to be implemented based on vector store capabilities
        # For now, return common namespaces
        return [self.config.default_namespace, "code_analysis", "research"]
    
    async def clear_user_cache(self, user_id: str) -> bool:
        """
        Löscht Cache für einen User
        """
        try:
            if self.response_cache:
                keys_to_remove = [
                    key for key, response in self.response_cache.items()
                    if response.user_id == user_id
                ]
                
                for key in keys_to_remove:
                    del self.response_cache[key]
                
                print(f"✅ Cleared {len(keys_to_remove)} cache entries for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error clearing user cache: {e}")
            return False
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """
        Optimiert Performance aller Components
        """
        optimization_results = {}
        
        try:
            if not self.initialized:
                await self.initialize()
            
            # Clear old cache entries
            if self.response_cache and len(self.response_cache) > self.max_cache_size * 0.8:
                old_size = len(self.response_cache)
                # Keep only most recent entries
                cache_items = list(self.response_cache.items())
                self.response_cache = dict(cache_items[-int(self.max_cache_size * 0.6):])
                optimization_results["cache_cleaned"] = old_size - len(self.response_cache)
            
            # Optimize embedding engine cache
            if self.embedding_engine:
                embedding_stats = self.embedding_engine.get_stats()
                if embedding_stats["cache_size"] > 800:
                    self.embedding_engine.clear_cache()
                    optimization_results["embedding_cache_cleared"] = True
            
            # Optimize LLM handler cache
            if self.llm_handler:
                llm_stats = self.llm_handler.get_performance_stats()
                if llm_stats["cache_hit_rate"] < 0.3:  # Low hit rate
                    self.llm_handler.clear_cache()
                    optimization_results["llm_cache_cleared"] = True
            
            optimization_results["status"] = "completed"
            optimization_results["timestamp"] = datetime.now().isoformat()
            
            print("🔧 Performance optimization completed")
            
            return optimization_results
            
        except Exception as e:
            print(f"❌ Performance optimization failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Liefert umfassende Performance-Statistiken
        """
        avg_processing_time = self.total_processing_time / self.total_queries if self.total_queries > 0 else 0
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        
        stats = {
            "orchestrator": {
                "total_queries": self.total_queries,
                "total_processing_time": self.total_processing_time,
                "average_processing_time": avg_processing_time,
                "cache_hit_rate": cache_hit_rate,
                "cache_size": len(self.response_cache) if self.response_cache else 0,
                "mode_usage": dict(self.mode_usage),
                "active_conversations": len(self.conversations)
            },
            "components": {
                "initialized": self.initialized,
                "status": self.component_status
            }
        }
        
        # Add component stats if available
        try:
            if self.embedding_engine:
                stats["embedding_engine"] = self.embedding_engine.get_stats()
            
            if self.vector_store:
                stats["vector_store"] = self.vector_store.get_performance_stats()
            
            if self.llm_handler:
                stats["llm_handler"] = self.llm_handler.get_performance_stats()
            
            if self.rerank_engine:
                stats["rerank_engine"] = self.rerank_engine.get_performance_stats()
            
            if self.document_processor:
                stats["document_processor"] = self.document_processor.get_performance_stats()
            
            if self.code_analyzer:
                stats["code_analyzer"] = self.code_analyzer.get_performance_stats()
            
            if self.research_assistant:
                stats["research_assistant"] = self.research_assistant.get_performance_stats()
                
        except Exception as e:
            stats["stats_error"] = str(e)
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Umfassender Health Check aller Components
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        try:
            if not self.initialized:
                await self.initialize()
            
            # Test embedding engine
            try:
                test_result = await self.embedding_engine.embed(["health check"])
                health_status["components"]["embedding_engine"] = {
                    "status": "healthy",
                    "dimension": len(test_result.embeddings[0]) if test_result.embeddings else 0
                }
            except Exception as e:
                health_status["components"]["embedding_engine"] = {"status": "unhealthy", "error": str(e)}
                health_status["overall_status"] = "degraded"
            
            # Test vector store
            try:
                # Try a simple operation
                await self.vector_store.create_user_store("health_check", "test")
                health_status["components"]["vector_store"] = {"status": "healthy"}
            except Exception as e:
                health_status["components"]["vector_store"] = {"status": "unhealthy", "error": str(e)}
                health_status["overall_status"] = "degraded"
            
            # Test LLM handler
            try:
                test_request = LLMRequest(
                    prompt="Test",
                    user_id="health_check",
                    max_tokens=5
                )
                test_response = await self.llm_handler.generate_single(test_request)
                health_status["components"]["llm_handler"] = {
                    "status": "healthy",
                    "response_length": len(test_response.response)
                }
            except Exception as e:
                health_status["components"]["llm_handler"] = {"status": "unhealthy", "error": str(e)}
                health_status["overall_status"] = "degraded"
            
            # Test rerank engine
            if self.rerank_engine:
                try:
                    test_candidates = [
                        {"id": "1", "content": "test content", "metadata": {}, "score": 0.8}
                    ]
                    await self.rerank_engine.rerank("test query", test_candidates, top_k=1)
                    health_status["components"]["rerank_engine"] = {"status": "healthy"}
                except Exception as e:
                    health_status["components"]["rerank_engine"] = {"status": "unhealthy", "error": str(e)}
            else:
                health_status["components"]["rerank_engine"] = {"status": "disabled"}
            
            return health_status
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "overall_status": "unhealthy",
                "error": str(e)
            }
    
    async def export_configuration(self, output_path: Union[str, Path]) -> bool:
        """
        Exportiert aktuelle Konfiguration
        """
        try:
            config_data = {
                "rag_config": asdict(self.config),
                "component_status": self.component_status,
                "performance_stats": self.get_comprehensive_stats(),
                "export_timestamp": datetime.now().isoformat()
            }
            
            output_path = Path(output_path)
            
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(config_data, indent=2, ensure_ascii=False))
            
            print(f"✅ Configuration exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Configuration export failed: {e}")
            return False
    
    async def benchmark(self) -> Dict[str, Any]:
        """
        Comprehensive Benchmark aller RAG Modi
        """
        print("🏁 Running comprehensive RAG Orchestrator benchmark...")
        
        test_queries = [
            RAGQuery("What is machine learning?", "benchmark_user", RAGMode.DOCUMENT_QA),
            RAGQuery("How do I implement a Python function?", "benchmark_user", RAGMode.CODE_ASSISTANCE),
            RAGQuery("Latest developments in AI", "benchmark_user", RAGMode.RESEARCH),
            RAGQuery("Tell me more about the previous topic", "benchmark_user", RAGMode.CONVERSATIONAL, conversation_id="benchmark_conv"),
            RAGQuery("Comprehensive overview of software development", "benchmark_user", RAGMode.HYBRID)
        ]
        
        benchmark_results = {}
        
        for query in test_queries:
            try:
                start_time = time.time()
                response = await self.query(query)
                query_time = time.time() - start_time
                
                benchmark_results[query.mode.value] = {
                    "processing_time": query_time,
                    "answer_length": len(response.answer),
                    "source_count": response.source_count,
                    "confidence": response.confidence,
                    "token_count": response.token_count,
                    "status": "success"
                }
                
            except Exception as e:
                benchmark_results[query.mode.value] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        # Overall metrics
        successful_queries = [r for r in benchmark_results.values() if r.get("status") == "success"]
        
        if successful_queries:
            benchmark_results["summary"] = {
                "total_queries": len(test_queries),
                "successful_queries": len(successful_queries),
                "average_processing_time": sum(r["processing_time"] for r in successful_queries) / len(successful_queries),
                "average_confidence": sum(r["confidence"] for r in successful_queries) / len(successful_queries),
                "total_tokens": sum(r["token_count"] for r in successful_queries),
                "success_rate": len(successful_queries) / len(test_queries) * 100
            }
        
        return benchmark_results
    
    async def close(self) -> None:
        """
        Schließt alle Resources
        """
        print("🔄 Shutting down RAG Orchestrator...")
        
        try:
            # Close vector store
            if self.vector_store:
                await self.vector_store.close()
            
            # Close research assistant
            if self.research_assistant:
                await self.research_assistant.close()
            
            # Clear caches
            if self.response_cache:
                self.response_cache.clear()
            
            self.conversations.clear()
            
            print("✅ RAG Orchestrator shutdown complete")
            
        except Exception as e:
            print(f"⚠️ Error during shutdown: {e}")

# Convenience Functions für häufige Use Cases
class RAGBuilder:
    """
    Builder Pattern für einfache RAG Orchestrator Setup
    """
    
    def __init__(self):
        self.config = RAGConfig()
    
    def with_models(self, embedding_model: str, llm_model: str) -> 'RAGBuilder':
        self.config.embedding_model = embedding_model
        self.config.llm_model = llm_model
        return self
    
    def with_vector_store(self, url: str, api_key: Optional[str] = None) -> 'RAGBuilder':
        self.config.vector_store_url = url
        self.config.vector_store_api_key = api_key
        return self
    
    def with_performance_settings(self, 
                                batch_size: int = 5, 
                                max_tokens: int = 1024, 
                                cache_enabled: bool = True) -> 'RAGBuilder':
        self.config.batch_size = batch_size
        self.config.max_tokens = max_tokens
        self.config.cache_enabled = cache_enabled
        return self
    
    def with_retrieval_settings(self, 
                              default_k: int = 5, 
                              rerank_enabled: bool = True) -> 'RAGBuilder':
        self.config.default_k = default_k
        self.config.rerank_enabled = rerank_enabled
        return self
    
    async def build(self) -> MLXRAGOrchestrator:
        orchestrator = MLXRAGOrchestrator(self.config)
        await orchestrator.initialize()
        return orchestrator

# Usage Examples
async def example_usage():
    """Beispiele für RAG Orchestrator Usage"""
    
    # Method 1: Direct initialization
    config = RAGConfig(
        embedding_model="mlx-community/gte-small",
        llm_model="mlx-community/gemma-2-9b-it-4bit",
        vector_store_url="http://localhost:8000",
        rerank_enabled=True,
        cache_enabled=True
    )
    
    orchestrator = MLXRAGOrchestrator(config)
    await orchestrator.initialize()
    
    try:
        # Add documents
        success = await orchestrator.add_document(
            "example.pdf",
            user_id="user_123",
            namespace="documents",
            metadata={"category": "research", "priority": "high"}
        )
        print(f"Document added: {success}")
        
        # Add code repository
        repo_success = await orchestrator.add_code_repository(
            "path/to/repository",
            user_id="user_123"
        )
        print(f"Repository added: {repo_success}")
        
        # Different query types
        queries = [
            # Document Q&A
            RAGQuery(
                query="What are the main findings in the research?",
                user_id="user_123",
                mode=RAGMode.DOCUMENT_QA,
                namespace="documents"
            ),
            
            # Code assistance
            RAGQuery(
                query="How do I implement error handling in this codebase?",
                user_id="user_123",
                mode=RAGMode.CODE_ASSISTANCE
            ),
            
            # Web research
            RAGQuery(
                query="What are the latest trends in machine learning?",
                user_id="user_123",
                mode=RAGMode.RESEARCH
            ),
            
            # Conversational
            RAGQuery(
                query="Can you explain that in more detail?",
                user_id="user_123",
                mode=RAGMode.CONVERSATIONAL,
                conversation_id="conv_123"
            ),
            
            # Hybrid approach
            RAGQuery(
                query="Compare machine learning frameworks and show code examples",
                user_id="user_123",
                mode=RAGMode.HYBRID
            )
        ]
        
        # Process single queries
        for query in queries:
            response = await orchestrator.query(query)
            print(f"\nQuery: {query.query}")
            print(f"Mode: {query.mode.value}")
            print(f"Answer: {response.answer[:200]}...")
            print(f"Sources: {response.source_count}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Time: {response.processing_time:.2f}s")
        
        # Batch processing
        batch_responses = await orchestrator.batch_query(queries)
        print(f"\nBatch processed {len(batch_responses)} queries")
        
        # Health check
        health = await orchestrator.health_check()
        print(f"\nHealth Status: {health['overall_status']}")
        
        # Performance stats
        stats = orchestrator.get_comprehensive_stats()
        print(f"\nTotal Queries: {stats['orchestrator']['total_queries']}")
        print(f"Cache Hit Rate: {stats['orchestrator']['cache_hit_rate']:.2%}")
        
        # Benchmark
        benchmark_results = await orchestrator.benchmark()
        print(f"\nBenchmark Results: {benchmark_results.get('summary', {})}")
        
        # Export configuration
        await orchestrator.export_configuration("rag_config.json")
        
    finally:
        await orchestrator.close()

# Method 2: Builder pattern
async def builder_example():
    """Beispiel mit Builder Pattern"""
    
    orchestrator = await (RAGBuilder()
                         .with_models("mlx-community/gte-large", "mlx-community/gemma-2-27b-it-4bit")
                         .with_vector_store("http://localhost:8000", "api-key")
                         .with_performance_settings(batch_size=10, cache_enabled=True)
                         .with_retrieval_settings(default_k=10, rerank_enabled=True)
                         .build())
    
    try:
        # Use orchestrator
        query = RAGQuery(
            query="Explain quantum computing",
            user_id="user_456",
            mode=RAGMode.HYBRID
        )
        
        response = await orchestrator.query(query)
        print(f"Response: {response.answer}")
        
    finally:
        await orchestrator.close()

if __name__ == "__main__":
    asyncio.run(example_usage())_id,
                custom_metadata=metadata
            )
            
            # Save to vector store
            namespace = namespace or self.config.default_namespace
            success = await self.document_processor.save_to_vector_store(
                processed_doc=processed_doc,
                vector_store=self.vector_store,
                user_id=user_id,
                model_id=self.config.embedding_model.split('/')[-1]
            )
            
            if success:
                print(f"✅ Document added: {Path(file_path).name}")
            
            return success
            
        except Exception as e:
            print(f"❌ Failed to add document: {e}")
            return False
    
    async def add_code_repository(self, 
                                repo_path: Union[str, Path], 
                                user_id: str) -> bool:
        """
        Analysiert und fügt Code Repository hinzu
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Analyze repository
            analysis = await self.code_analyzer.analyze_repository(
                repo_path=repo_path,
                user_id=user_id
            )
            
            # Save to vector store
            success = await self.code_analyzer.save_to_vector_store(
                analysis=analysis,
                vector_store=self.vector_store,
                user_id=user