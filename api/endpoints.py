"""
Enhanced API Endpoints für mlx-langchain-lite
Erweitert den bestehenden API-Layer mit zusätzlichen Batch-APIs und Multi-Agent Koordination
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, BackgroundTasks, Depends, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Enhanced imports
from tools.document_processor import DocumentProcessor, DocumentConfig, ProcessingResult
from tools.code_analyzer import CodeAnalyzer, CodeAnalysisConfig, RepositoryStructure
from tools.research_assistant import ResearchAssistant, ResearchConfig, ResearchQuery, ResearchResult
from mlx_components.rag_orchestrator import RAGOrchestrator

logger = logging.getLogger(__name__)

# Enhanced Pydantic Models
class MultiAgentRequest(BaseModel):
    """Request for multi-agent coordination"""
    query: str
    user_id: str
    session_id: Optional[str] = None
    agents: List[str] = Field(default=["rag", "research"], description="Agents to involve")
    coordination_strategy: str = Field(default="parallel", description="parallel, sequential, or adaptive")
    max_execution_time: int = Field(default=120, description="Max execution time in seconds")
    priority: str = Field(default="normal", description="normal, high, or urgent")

class MultiAgentResponse(BaseModel):
    """Response from multi-agent coordination"""
    query: str
    session_id: str
    agents_used: List[str]
    coordination_strategy: str
    responses: Dict[str, Any]
    synthesis: str
    confidence_score: float
    execution_time: float
    metadata: Dict[str, Any]

class BatchDocumentRequest(BaseModel):
    """Request for batch document processing"""
    user_id: str
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    notify_webhook: Optional[str] = None
    priority: str = Field(default="normal")

class BatchDocumentResponse(BaseModel):
    """Response from batch document processing"""
    batch_id: str
    user_id: str
    total_files: int
    processed_files: int
    failed_files: int
    processing_time: float
    results: List[ProcessingResult]
    status: str
    errors: List[str] = Field(default_factory=list)

class CodeAnalysisRequest(BaseModel):
    """Request for code analysis"""
    repository_path: str
    user_id: str
    analysis_types: List[str] = Field(default=["structure", "complexity", "dependencies"])
    languages: Optional[List[str]] = None
    include_embeddings: bool = True

class CodeAnalysisResponse(BaseModel):
    """Response from code analysis"""
    analysis_id: str
    repository_path: str
    user_id: str
    analysis_types: List[str]
    structure: RepositoryStructure
    insights: Dict[str, Any]
    recommendations: List[str]
    processing_time: float

class ResearchTaskRequest(BaseModel):
    """Request for research task"""
    query: str
    user_id: str
    research_type: str = Field(default="comprehensive", description="quick, comprehensive, or deep")
    sources: Optional[List[str]] = None
    time_constraint: Optional[str] = None
    language: str = "en"
    max_sources: int = 20

class ResearchTaskResponse(BaseModel):
    """Response from research task"""
    task_id: str
    query: str
    user_id: str
    result: ResearchResult
    cached: bool = False
    processing_time: float

class BatchProcessingStatus(BaseModel):
    """Status of batch processing"""
    batch_id: str
    status: str  # pending, processing, completed, failed
    progress: float  # 0.0 to 1.0
    total_items: int
    processed_items: int
    failed_items: int
    started_at: datetime
    estimated_completion: Optional[datetime] = None
    current_operation: str
    errors: List[str] = Field(default_factory=list)

class SystemMetrics(BaseModel):
    """Enhanced system metrics"""
    timestamp: datetime
    mlx_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    agent_metrics: Dict[str, Any]
    resource_usage: Dict[str, Any]
    batch_processing: Dict[str, Any]

# Enhanced API State Management
class EnhancedAPIState:
    """Enhanced API state with batch processing and multi-agent coordination"""
    
    def __init__(self):
        # Core components
        self.rag_orchestrator: Optional[RAGOrchestrator] = None
        self.document_processor: Optional[DocumentProcessor] = None
        self.code_analyzer: Optional[CodeAnalyzer] = None
        self.research_assistant: Optional[ResearchAssistant] = None
        
        # Connection management
        self.active_connections: Dict[str, WebSocket] = {}
        self.agent_connections: Dict[str, Dict[str, WebSocket]] = {}
        
        # Batch processing
        self.batch_jobs: Dict[str, Dict[str, Any]] = {}
        self.batch_status: Dict[str, BatchProcessingStatus] = {}
        
        # Multi-agent coordination
        self.active_agents: Dict[str, bool] = {
            "rag": True,
            "research": True,
            "code_analyzer": True,
            "document_processor": True
        }
        
        # Performance tracking
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.agent_usage_stats = {agent: 0 for agent in self.active_agents}
        self.response_times = []
        
        # Background tasks
        self.background_tasks: Dict[str, asyncio.Task] = {}

enhanced_api_state = EnhancedAPIState()

# Authentication (simplified)
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Get current user from token (simplified implementation)"""
    if not credentials:
        return "anonymous"
    # In production, validate JWT token here
    return "authenticated_user"

# Enhanced API Endpoints

# Multi-Agent Coordination Endpoints
@app.post("/agents/coordinate", response_model=MultiAgentResponse)
async def coordinate_agents(
    request: MultiAgentRequest,
    current_user: str = Depends(get_current_user)
):
    """Coordinate multiple agents for complex queries"""
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        logger.info(f"Starting multi-agent coordination for query: {request.query[:100]}...")
        
        agent_responses = {}
        agents_used = []
        
        if request.coordination_strategy == "parallel":
            # Execute agents in parallel
            tasks = []
            
            if "rag" in request.agents and enhanced_api_state.active_agents["rag"]:
                task = _execute_rag_agent(request.query, request.user_id, session_id)
                tasks.append(("rag", task))
            
            if "research" in request.agents and enhanced_api_state.active_agents["research"]:
                task = _execute_research_agent(request.query, request.user_id, session_id)
                tasks.append(("research", task))
            
            if "code_analyzer" in request.agents and enhanced_api_state.active_agents["code_analyzer"]:
                # Only if query seems code-related
                if any(keyword in request.query.lower() for keyword in ['code', 'function', 'class', 'repository', 'programming']):
                    task = _execute_code_agent(request.query, request.user_id)
                    tasks.append(("code_analyzer", task))
            
            # Execute all tasks
            if tasks:
                task_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                
                for i, (agent_name, _) in enumerate(tasks):
                    result = task_results[i]
                    if isinstance(result, Exception):
                        logger.error(f"Agent {agent_name} failed: {result}")
                        agent_responses[agent_name] = {"error": str(result)}
                    else:
                        agent_responses[agent_name] = result
                        agents_used.append(agent_name)
                        enhanced_api_state.agent_usage_stats[agent_name] += 1
        
        elif request.coordination_strategy == "sequential":
            # Execute agents sequentially, using previous results
            context = request.query
            
            for agent in request.agents:
                if agent in enhanced_api_state.active_agents and enhanced_api_state.active_agents[agent]:
                    try:
                        if agent == "rag":
                            result = await _execute_rag_agent(context, request.user_id, session_id)
                        elif agent == "research":
                            result = await _execute_research_agent(context, request.user_id, session_id)
                        elif agent == "code_analyzer":
                            result = await _execute_code_agent(context, request.user_id)
                        else:
                            continue
                        
                        agent_responses[agent] = result
                        agents_used.append(agent)
                        enhanced_api_state.agent_usage_stats[agent] += 1
                        
                        # Update context for next agent
                        if isinstance(result, dict) and "response" in result:
                            context = f"{context}\n\nPrevious analysis: {result['response']}"
                            
                    except Exception as e:
                        logger.error(f"Sequential agent {agent} failed: {e}")
                        agent_responses[agent] = {"error": str(e)}
        
        # Synthesize responses
        synthesis = await _synthesize_agent_responses(agent_responses, request.query)
        confidence_score = _calculate_confidence_score(agent_responses)
        
        execution_time = time.time() - start_time
        enhanced_api_state.request_count += 1
        enhanced_api_state.response_times.append(execution_time)
        
        return MultiAgentResponse(
            query=request.query,
            session_id=session_id,
            agents_used=agents_used,
            coordination_strategy=request.coordination_strategy,
            responses=agent_responses,
            synthesis=synthesis,
            confidence_score=confidence_score,
            execution_time=execution_time,
            metadata={
                "total_agents_requested": len(request.agents),
                "successful_agents": len(agents_used),
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        enhanced_api_state.error_count += 1
        logger.error(f"Multi-agent coordination failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _execute_rag_agent(query: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """Execute RAG agent"""
    if not enhanced_api_state.rag_orchestrator:
        raise ValueError("RAG orchestrator not initialized")
    
    result = await enhanced_api_state.rag_orchestrator.process_query(
        query=query,
        user_id=user_id,
        session_id=session_id,
        max_tokens=512,
        temperature=0.7
    )
    
    return {
        "agent": "rag",
        "response": result["response"],
        "sources": result["sources"],
        "confidence": result.get("confidence", 0.8)
    }

async def _execute_research_agent(query: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """Execute research agent"""
    if not enhanced_api_state.research_assistant:
        raise ValueError("Research assistant not initialized")
    
    research_query = ResearchQuery(
        query=query,
        user_id=user_id,
        session_id=session_id,
        query_type="factual"
    )
    
    result = await enhanced_api_state.research_assistant.research(research_query)
    
    return {
        "agent": "research",
        "response": result.summary,
        "sources": [{"url": source.url, "title": source.title} for source in result.sources[:5]],
        "confidence": result.confidence_score,
        "key_findings": result.key_findings
    }

async def _execute_code_agent(query: str, user_id: str) -> Dict[str, Any]:
    """Execute code analysis agent"""
    if not enhanced_api_state.code_analyzer:
        raise ValueError("Code analyzer not initialized")
    
    # Search for code elements related to query
    results = await enhanced_api_state.code_analyzer.search_code_elements(query, user_id, top_k=5)
    
    return {
        "agent": "code_analyzer",
        "response": f"Found {len(results)} relevant code elements",
        "code_elements": results,
        "confidence": 0.9 if results else 0.3
    }

async def _synthesize_agent_responses(agent_responses: Dict[str, Any], original_query: str) -> str:
    """Synthesize responses from multiple agents"""
    if not agent_responses:
        return "No agent responses to synthesize."
    
    synthesis_parts = []
    synthesis_parts.append(f"Comprehensive analysis for: {original_query}\n")
    
    for agent, response in agent_responses.items():
        if "error" in response:
            synthesis_parts.append(f"⚠️ {agent.upper()} Agent: Encountered an error")
            continue
        
        agent_response = response.get("response", "No response")
        confidence = response.get("confidence", 0.5)
        
        synthesis_parts.append(f"\n🤖 {agent.upper()} Agent (Confidence: {confidence:.1%}):")
        synthesis_parts.append(agent_response)
        
        # Add specific information from each agent
        if agent == "research" and "key_findings" in response:
            synthesis_parts.append("Key Research Findings:")
            for finding in response["key_findings"][:3]:
                synthesis_parts.append(f"• {finding}")
        
        elif agent == "code_analyzer" and "code_elements" in response:
            synthesis_parts.append("Relevant Code Elements:")
            for element in response["code_elements"][:3]:
                name = element.get("metadata", {}).get("name", "Unknown")
                synthesis_parts.append(f"• {name}")
    
    # Add summary
    synthesis_parts.append("\n🔍 SYNTHESIS:")
    synthesis_parts.append("Based on the analysis from multiple AI agents, this comprehensive response combines ")
    synthesis_parts.append("information retrieval, web research, and code analysis to provide a well-rounded answer.")
    
    return "\n".join(synthesis_parts)

def _calculate_confidence_score(agent_responses: Dict[str, Any]) -> float:
    """Calculate overall confidence score"""
    if not agent_responses:
        return 0.0
    
    confidences = []
    for response in agent_responses.values():
        if "error" not in response:
            confidences.append(response.get("confidence", 0.5))
    
    if not confidences:
        return 0.0
    
    # Weighted average with bonus for multiple successful agents
    avg_confidence = sum(confidences) / len(confidences)
    agent_bonus = min(0.2, (len(confidences) - 1) * 0.1)
    
    return min(1.0, avg_confidence + agent_bonus)

# Batch Document Processing Endpoints
@app.post("/documents/batch/upload", response_model=BatchDocumentResponse)
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    request_data: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: str = Depends(get_current_user)
):
    """Upload and process multiple documents in batch"""
    try:
        request = BatchDocumentRequest.parse_raw(request_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {e}")
    
    batch_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Initialize batch status
    enhanced_api_state.batch_status[batch_id] = BatchProcessingStatus(
        batch_id=batch_id,
        status="processing",
        progress=0.0,
        total_items=len(files),
        processed_items=0,
        failed_items=0,
        started_at=datetime.now(),
        current_operation="uploading_files"
    )
    
    try:
        # Save uploaded files temporarily
        temp_files = []
        upload_dir = Path("temp_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        for file in files:
            if file.filename:
                temp_path = upload_dir / f"{batch_id}_{file.filename}"
                content = await file.read()
                with open(temp_path, "wb") as f:
                    f.write(content)
                temp_files.append(str(temp_path))
        
        # Process documents in background
        background_tasks.add_task(
            _process_documents_batch,
            batch_id,
            temp_files,
            request.user_id,
            request.processing_options,
            request.notify_webhook
        )
        
        # Return immediate response
        return BatchDocumentResponse(
            batch_id=batch_id,
            user_id=request.user_id,
            total_files=len(files),
            processed_files=0,
            failed_files=0,
            processing_time=0.0,
            results=[],
            status="processing"
        )
        
    except Exception as e:
        enhanced_api_state.batch_status[batch_id].status = "failed"
        enhanced_api_state.batch_status[batch_id].errors.append(str(e))
        logger.error(f"Batch upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _process_documents_batch(
    batch_id: str,
    file_paths: List[str],
    user_id: str,
    processing_options: Dict[str, Any],
    notify_webhook: Optional[str]
):
    """Process documents in batch (background task)"""
    try:
        status = enhanced_api_state.batch_status[batch_id]
        status.current_operation = "processing_documents"
        
        if not enhanced_api_state.document_processor:
            raise ValueError("Document processor not initialized")
        
        # Process documents
        results = await enhanced_api_state.document_processor.process_documents_batch(file_paths, user_id)
        
        # Update status
        processed_count = sum(1 for r in results if not r.errors)
        failed_count = len(results) - processed_count
        
        status.processed_items = processed_count
        status.failed_items = failed_count
        status.progress = 1.0
        status.status = "completed" if failed_count == 0 else "partial_failure"
        status.current_operation = "completed"
        
        # Store results
        enhanced_api_state.batch_jobs[batch_id] = {
            "results": results,
            "completed_at": datetime.now(),
            "user_id": user_id
        }
        
        # Clean up temp files
        for file_path in file_paths:
            try:
                Path(file_path).unlink(missing_ok=True)
            except:
                pass
        
        # Send webhook notification if provided
        if notify_webhook:
            await _send_webhook_notification(notify_webhook, batch_id, status)
        
        logger.info(f"Batch {batch_id} completed: {processed_count} processed, {failed_count} failed")
        
    except Exception as e:
        status = enhanced_api_state.batch_status[batch_id]
        status.status = "failed"
        status.errors.append(str(e))
        logger.error(f"Batch processing failed for {batch_id}: {e}")

@app.get("/documents/batch/{batch_id}/status", response_model=BatchProcessingStatus)
async def get_batch_status(
    batch_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get status of batch processing"""
    if batch_id not in enhanced_api_state.batch_status:
        raise HTTPException(status_code=404, detail="Batch not found")
    
    return enhanced_api_state.batch_status[batch_id]

@app.get("/documents/batch/{batch_id}/results", response_model=BatchDocumentResponse)
async def get_batch_results(
    batch_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get results of batch processing"""
    if batch_id not in enhanced_api_state.batch_jobs:
        raise HTTPException(status_code=404, detail="Batch results not found")
    
    job_data = enhanced_api_state.batch_jobs[batch_id]
    status = enhanced_api_state.batch_status[batch_id]
    
    return BatchDocumentResponse(
        batch_id=batch_id,
        user_id=job_data["user_id"],
        total_files=status.total_items,
        processed_files=status.processed_items,
        failed_files=status.failed_items,
        processing_time=(job_data["completed_at"] - status.started_at).total_seconds(),
        results=job_data["results"],
        status=status.status,
        errors=status.errors
    )

# Code Analysis Endpoints
@app.post("/code/analyze", response_model=CodeAnalysisResponse)
async def analyze_code_repository(
    request: CodeAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Analyze code repository"""
    if not enhanced_api_state.code_analyzer:
        raise HTTPException(status_code=503, detail="Code analyzer not available")
    
    analysis_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Analyze repository
        structure = await enhanced_api_state.code_analyzer.analyze_repository(
            request.repository_path,
            request.user_id
        )
        
        # Generate insights
        insights = _generate_code_insights(structure, request.analysis_types)
        recommendations = _generate_code_recommendations(structure, insights)
        
        processing_time = time.time() - start_time
        
        return CodeAnalysisResponse(
            analysis_id=analysis_id,
            repository_path=request.repository_path,
            user_id=request.user_id,
            analysis_types=request.analysis_types,
            structure=structure,
            insights=insights,
            recommendations=recommendations,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Code analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_code_insights(structure: RepositoryStructure, analysis_types: List[str]) -> Dict[str, Any]:
    """Generate insights from code analysis"""
    insights = {}
    
    if "structure" in analysis_types:
        insights["structure"] = {
            "total_files": structure.total_files,
            "total_lines": structure.total_lines,
            "languages": structure.languages,
            "largest_language": max(structure.languages.items(), key=lambda x: x[1])[0] if structure.languages else None,
            "modules_count": len(structure.modules),
            "classes_count": len(structure.classes),
            "functions_count": len(structure.functions)
        }
    
    if "complexity" in analysis_types:
        insights["complexity"] = {
            "average_complexity": structure.complexity_stats.get("avg", 0),
            "max_complexity": structure.complexity_stats.get("max", 0),
            "high_complexity_functions": len([
                f for f in structure.functions 
                if f.complexity and f.complexity > 10
            ])
        }
    
    if "dependencies" in analysis_types:
        insights["dependencies"] = {
            "dependency_count": len(structure.dependency_graph),
            "most_dependent": max(
                structure.dependency_graph.items(), 
                key=lambda x: len(x[1])
            )[0] if structure.dependency_graph else None,
            "circular_dependencies": _detect_circular_dependencies(structure.dependency_graph)
        }
    
    return insights

def _generate_code_recommendations(structure: RepositoryStructure, insights: Dict[str, Any]) -> List[str]:
    """Generate code recommendations"""
    recommendations = []
    
    # Complexity recommendations
    if "complexity" in insights:
        if insights["complexity"]["high_complexity_functions"] > 0:
            recommendations.append(
                f"Consider refactoring {insights['complexity']['high_complexity_functions']} "
                "high-complexity functions (>10 cyclomatic complexity)"
            )
        
        if insights["complexity"]["max_complexity"] > 20:
            recommendations.append("Review the most complex function - consider breaking it down")
    
    # Structure recommendations
    if "structure" in insights:
        if insights["structure"]["functions_count"] / max(insights["structure"]["classes_count"], 1) > 10:
            recommendations.append("Consider organizing standalone functions into classes")
        
        if structure.total_lines / structure.total_files > 500:
            recommendations.append("Consider splitting large files into smaller modules")
    
    # Language diversity
    if len(structure.languages) > 3:
        recommendations.append("Multiple programming languages detected - ensure consistent practices")
    
    return recommendations

def _detect_circular_dependencies(dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
    """Detect circular dependencies (simplified)"""
    # Simple cycle detection - in practice, use more sophisticated algorithms
    circular = []
    
    for node, deps in dependency_graph.items():
        for dep in deps:
            if dep in dependency_graph and node in dependency_graph[dep]:
                if [dep, node] not in circular and [node, dep] not in circular:
                    circular.append([node, dep])
    
    return circular

# Research Task Endpoints
@app.post("/research/task", response_model=ResearchTaskResponse)
async def create_research_task(
    request: ResearchTaskRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
):
    """Create and execute research task"""
    if not enhanced_api_state.research_assistant:
        raise HTTPException(status_code=503, detail="Research assistant not available")
    
    task_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Check for cached results first
        cached_result = await _check_research_cache(request.query, request.user_id)
        
        if cached_result:
            return ResearchTaskResponse(
                task_id=task_id,
                query=request.query,
                user_id=request.user_id,
                result=cached_result,
                cached=True,
                processing_time=time.time() - start_time
            )
        
        # Create research query
        research_query = ResearchQuery(
            query=request.query,
            user_id=request.user_id,
            session_id=task_id,
            query_type=request.research_type,
            time_constraint=request.time_constraint,
            language=request.language
        )
        
        # Execute research
        result = await enhanced_api_state.research_assistant.research(research_query)
        
        processing_time = time.time() - start_time
        
        return ResearchTaskResponse(
            task_id=task_id,
            query=request.query,
            user_id=request.user_id,
            result=result,
            cached=False,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Research task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _check_research_cache(query: str, user_id: str) -> Optional[ResearchResult]:
    """Check for cached research results"""
    try:
        if enhanced_api_state.research_assistant:
            # Search for similar previous research
            similar_research = await enhanced_api_state.research_assistant.search_previous_research(
                query, user_id, top_k=1
            )
            
            if similar_research and similar_research[0].get("score", 0) > 0.9:
                # Return cached result if very similar and recent (< 24 hours)
                metadata = similar_research[0].get("metadata", {})
                if time.time() - metadata.get("timestamp", 0) < 86400:  # 24 hours
                    # Reconstruct ResearchResult from cached data
                    # This is a simplified version - in practice, you'd store the full result
                    pass
        
        return None
        
    except Exception as e:
        logger.warning(f"Cache check failed: {e}")
        return None

# Enhanced Monitoring and Analytics
@app.get("/admin/metrics/enhanced", response_model=SystemMetrics)
async def get_enhanced_metrics():
    """Get comprehensive system metrics"""
    import psutil
    import mlx.core as mx
    
    # MLX metrics
    memory_info = mx.metal.get_memory_info()
    mlx_metrics = {
        "memory_total_gb": memory_info["total"] / (1024**3),
        "memory_available_gb": memory_info["available"] / (1024**3),
        "memory_usage_percent": (memory_info["total"] - memory_info["available"]) / memory_info["total"] * 100
    }
    
    # Performance metrics
    performance_metrics = {
        "total_requests": enhanced_api_state.request_count,
        "error_count": enhanced_api_state.error_count,
        "error_rate": enhanced_api_state.error_count / max(enhanced_api_state.request_count, 1),
        "average_response_time": sum(enhanced_api_state.response_times[-100:]) / len(enhanced_api_state.response_times[-100:]) if enhanced_api_state.response_times else 0,
        "uptime_seconds": time.time() - enhanced_api_state.start_time
    }
    
    # Agent metrics
    agent_metrics = {
        "active_agents": enhanced_api_state.active_agents,
        "agent_usage_stats": enhanced_api_state.agent_usage_stats,
        "active_connections": len(enhanced_api_state.active_connections),
        "background_tasks": len(enhanced_api_state.background_tasks)
    }
    
    # Resource usage
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    resource_usage = {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "disk_usage_percent": disk.percent,
        "disk_free_gb": disk.free / (1024**3)
    }
    
    # Batch processing stats
    batch_stats = {
        "active_batches": sum(1 for status in enhanced_api_state.batch_status.values() if status.status == "processing"),
        "completed_batches": sum(1 for status in enhanced_api_state.batch_status.values() if status.status == "completed"),
        "failed_batches": sum(1 for status in enhanced_api_state.batch_status.values() if status.status == "failed"),
        "total_batch_jobs": len(enhanced_api_state.batch_status)
    }
    
    return SystemMetrics(
        timestamp=datetime.now(),
        mlx_metrics=mlx_metrics,
        performance_metrics=performance_metrics,
        agent_metrics=agent_metrics,
        resource_usage=resource_usage,
        batch_processing=batch_stats
    )

# WebSocket for Multi-Agent Streaming
@app.websocket("/agents/stream")
async def websocket_multi_agent_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming multi-agent responses"""
    await websocket.accept()
    connection_id = f"agent_stream_{int(time.time() * 1000)}"
    enhanced_api_state.active_connections[connection_id] = websocket
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            try:
                request = MultiAgentRequest(**request_data)
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "error": f"Invalid request: {e}"
                }))
                continue
            
            # Stream multi-agent response
            await _stream_multi_agent_response(websocket, request)
            
    except WebSocketDisconnect:
        logger.info(f"Multi-agent WebSocket {connection_id} disconnected")
    except Exception as e:
        logger.error(f"Multi-agent WebSocket error: {e}")
    finally:
        if connection_id in enhanced_api_state.active_connections:
            del enhanced_api_state.active_connections[connection_id]

async def _stream_multi_agent_response(websocket: WebSocket, request: MultiAgentRequest):
    """Stream multi-agent coordination response"""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": "Starting multi-agent coordination",
            "session_id": session_id,
            "agents_requested": request.agents
        }))
        
        agent_responses = {}
        
        # Execute agents and stream results
        for agent in request.agents:
            if agent not in enhanced_api_state.active_agents or not enhanced_api_state.active_agents[agent]:
                continue
            
            await websocket.send_text(json.dumps({
                "type": "agent_start",
                "agent": agent,
                "message": f"Starting {agent} agent"
            }))
            
            try:
                if agent == "rag":
                    result = await _execute_rag_agent(request.query, request.user_id, session_id)
                elif agent == "research":
                    result = await _execute_research_agent(request.query, request.user_id, session_id)
                elif agent == "code_analyzer":
                    result = await _execute_code_agent(request.query, request.user_id)
                
                agent_responses[agent] = result
                
                await websocket.send_text(json.dumps({
                    "type": "agent_result",
                    "agent": agent,
                    "result": result
                }))
                
            except Exception as e:
                error_msg = f"Agent {agent} failed: {str(e)}"
                await websocket.send_text(json.dumps({
                    "type": "agent_error",
                    "agent": agent,
                    "error": error_msg
                }))
        
        # Send synthesis
        synthesis = await _synthesize_agent_responses(agent_responses, request.query)
        confidence = _calculate_confidence_score(agent_responses)
        
        await websocket.send_text(json.dumps({
            "type": "synthesis",
            "synthesis": synthesis,
            "confidence_score": confidence,
            "session_id": session_id
        }))
        
        # Send completion
        await websocket.send_text(json.dumps({
            "type": "complete",
            "session_id": session_id
        }))
        
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e)
        }))

# Utility Functions
async def _send_webhook_notification(webhook_url: str, batch_id: str, status: BatchProcessingStatus):
    """Send webhook notification for batch completion"""
    try:
        import aiohttp
        
        payload = {
            "batch_id": batch_id,
            "status": status.status,
            "progress": status.progress,
            "processed_items": status.processed_items,
            "failed_items": status.failed_items,
            "timestamp": datetime.now().isoformat()
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload, timeout=10) as response:
                if response.status == 200:
                    logger.info(f"Webhook notification sent for batch {batch_id}")
                else:
                    logger.warning(f"Webhook notification failed for batch {batch_id}: {response.status}")
                    
    except Exception as e:
        logger.error(f"Failed to send webhook notification: {e}")

# Enhanced lifespan management
@asynccontextmanager
async def enhanced_lifespan(app: FastAPI):
    """Enhanced application lifespan with all components"""
    # Startup
    logger.info("Initializing enhanced MLX-based RAG system...")
    
    try:
        # Initialize all components (reuse from original api_layer.py)
        from mlx_components.embedding_engine import EmbeddingEngine, EmbeddingConfig
        from mlx_components.vector_store import VectorStore, VectorStoreConfig
        from mlx_components.llm_handler import LLMHandler, LLMConfig
        from mlx_components.rag_orchestrator import RAGOrchestrator, RAGConfig
        
        # Initialize configurations
        embedding_config = EmbeddingConfig(
            model_path="mlx-community/all-MiniLM-L6-v2",
            max_batch_size=32,
            cache_embeddings=True
        )
        
        vector_config = VectorStoreConfig(
            base_url="http://localhost:8080",
            collection_name="theseus_rag",
            dimension=384,
            enable_multi_user=True
        )
        
        llm_config = LLMConfig(
            model_path="mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            max_tokens=2048,
            temperature=0.7,
            batch_size=8,
            enable_streaming=True,
            enable_caching=True
        )
        
        rag_config = RAGConfig(
            embedding_config=embedding_config,
            vector_config=vector_config,
            llm_config=llm_config,
            chunk_size=512,
            chunk_overlap=50,
            top_k=5,
            enable_reranking=True
        )
        
        # Initialize core components
        embedding_engine = EmbeddingEngine(embedding_config)
        vector_store = VectorStore(vector_config)
        llm_handler = LLMHandler(llm_config)
        
        await embedding_engine.initialize()
        await vector_store.initialize()
        await llm_handler.initialize()
        
        # Initialize RAG orchestrator
        enhanced_api_state.rag_orchestrator = RAGOrchestrator(rag_config)
        await enhanced_api_state.rag_orchestrator.initialize()
        
        # Initialize enhanced components
        doc_config = DocumentConfig(
            chunk_size=512,
            chunk_overlap=50,
            batch_size=32,
            enable_pii_filtering=True,
            enable_metadata_extraction=True
        )
        enhanced_api_state.document_processor = DocumentProcessor(doc_config, embedding_engine, vector_store)
        
        code_config = CodeAnalysisConfig(
            batch_size=32,
            enable_complexity_analysis=True,
            enable_semantic_analysis=True
        )
        enhanced_api_state.code_analyzer = CodeAnalyzer(code_config, embedding_engine, vector_store)
        
        research_config = ResearchConfig(
            max_search_results=20,
            batch_size=8,
            enable_fact_checking=True
        )
        enhanced_api_state.research_assistant = ResearchAssistant(research_config, llm_handler, embedding_engine, vector_store)
        
        logger.info("Enhanced RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced RAG system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down enhanced system...")
    if enhanced_api_state.rag_orchestrator:
        await enhanced_api_state.rag_orchestrator.cleanup()
    if enhanced_api_state.document_processor:
        await enhanced_api_state.document_processor.cleanup()
    if enhanced_api_state.code_analyzer:
        await enhanced_api_state.code_analyzer.cleanup()
    if enhanced_api_state.research_assistant:
        await enhanced_api_state.research_assistant.cleanup()

# Create enhanced app instance
enhanced_app = FastAPI(
    title="Theseus-TeamMind Enhanced RAG API",
    description="MLX-based RAG system with Multi-Agent coordination and Batch processing",
    version="2.0.0",
    lifespan=enhanced_lifespan
)

# Add CORS middleware
enhanced_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all endpoints
for route in app.routes:
    enhanced_app.routes.append(route)

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_api_endpoints:enhanced_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )