AgentType.CODE_ANALYZER: AgentCapability(
                agent_type=AgentType.CODE_ANALYZER,
                capabilities=["code_analysis", "complexity_assessment", "dependency_mapping", "pattern_detection"],
                input_types=["code_repository", "code_snippet", "file_path"],
                output_types=["analysis_report", "complexity_metrics", "recommendations"],
                processing_time_estimate=10.0,
                resource_requirements={"memory_mb": 800, "compute_score": 0.8}
            ),
            
            AgentType.DOCUMENT_PROCESSOR: AgentCapability(
                agent_type=AgentType.DOCUMENT_PROCESSOR,
                capabilities=["document_parsing", "content_extraction", "metadata_analysis", "chunking"],
                input_types=["pdf", "docx", "txt", "html", "md"],
                output_types=["extracted_text", "chunks", "metadata", "embeddings"],
                processing_time_estimate=5.0,
                resource_requirements={"memory_mb": 400, "compute_score": 0.6}
            )
        }
        
        # Initialize stats
        for agent_type in AgentType:
            self.agent_stats[agent_type] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "failed_tasks": 0,
                "avg_processing_time": 0.0,
                "avg_confidence": 0.0,
                "last_used": None
            }
    
    def register_agent(self, agent_type: AgentType, agent_instance: Any):
        """Register an agent instance"""
        self.agents[agent_type] = agent_instance
        self.agent_status[agent_type] = "ready"
        logger.info(f"Registered agent: {agent_type.value}")
    
    def is_agent_available(self, agent_type: AgentType) -> bool:
        """Check if agent is available"""
        return (agent_type in self.agents and 
                self.agent_status.get(agent_type) == "ready")
    
    def get_agent_capability(self, agent_type: AgentType) -> Optional[AgentCapability]:
        """Get agent capability"""
        return self.agent_capabilities.get(agent_type)
    
    def update_agent_stats(self, agent_type: AgentType, response: AgentResponse):
        """Update agent statistics"""
        stats = self.agent_stats[agent_type]
        stats["total_tasks"] += 1
        
        if response.success:
            stats["successful_tasks"] += 1
            # Update averages
            n = stats["successful_tasks"]
            stats["avg_processing_time"] = (
                (stats["avg_processing_time"] * (n-1) + response.processing_time) / n
            )
            stats["avg_confidence"] = (
                (stats["avg_confidence"] * (n-1) + response.confidence_score) / n
            )
        else:
            stats["failed_tasks"] += 1
        
        stats["last_used"] = datetime.now()

class TaskAnalyzer:
    """Analyzes tasks to determine optimal agent coordination"""
    
    def __init__(self, agent_manager: AgentManager):
        self.agent_manager = agent_manager
        
        # Task patterns and their optimal agent combinations
        self.task_patterns = {
            "document_question": [AgentType.RAG, AgentType.DOCUMENT_PROCESSOR],
            "research_query": [AgentType.RESEARCH, AgentType.RAG],
            "code_analysis": [AgentType.CODE_ANALYZER, AgentType.RAG],
            "comprehensive_analysis": [AgentType.RAG, AgentType.RESEARCH, AgentType.CODE_ANALYZER],
            "document_processing": [AgentType.DOCUMENT_PROCESSOR, AgentType.RAG],
            "fact_checking": [AgentType.RESEARCH, AgentType.RAG]
        }
    
    def analyze_task(self, task: Task) -> Dict[str, Any]:
        """Analyze task to determine optimal coordination strategy"""
        analysis = {
            "task_type": self._classify_task_type(task),
            "complexity": self._assess_complexity(task),
            "recommended_agents": [],
            "coordination_strategy": CoordinationStrategy.PARALLEL,
            "estimated_time": 0.0,
            "resource_requirements": {}
        }
        
        # Determine task type and recommended agents
        task_type = analysis["task_type"]
        if task_type in self.task_patterns:
            recommended_agents = self.task_patterns[task_type]
        else:
            recommended_agents = self._infer_agents_from_query(task.query)
        
        # Filter available agents
        available_agents = [
            agent for agent in recommended_agents 
            if self.agent_manager.is_agent_available(agent)
        ]
        
        analysis["recommended_agents"] = available_agents
        
        # Determine coordination strategy based on complexity and agents
        if len(available_agents) == 1:
            analysis["coordination_strategy"] = CoordinationStrategy.SEQUENTIAL
        elif analysis["complexity"] > 0.7:
            analysis["coordination_strategy"] = CoordinationStrategy.ADAPTIVE
        elif task.priority in [TaskPriority.HIGH, TaskPriority.URGENT]:
            analysis["coordination_strategy"] = CoordinationStrategy.PARALLEL
        else:
            analysis["coordination_strategy"] = CoordinationStrategy.SEQUENTIAL
        
        # Estimate processing time and resources
        total_time = 0.0
        total_resources = {"memory_mb": 0, "compute_score": 0}
        
        for agent_type in available_agents:
            capability = self.agent_manager.get_agent_capability(agent_type)
            if capability:
                if analysis["coordination_strategy"] == CoordinationStrategy.PARALLEL:
                    total_time = max(total_time, capability.processing_time_estimate)
                else:
                    total_time += capability.processing_time_estimate
                
                total_resources["memory_mb"] += capability.resource_requirements.get("memory_mb", 0)
                total_resources["compute_score"] = max(
                    total_resources["compute_score"], 
                    capability.resource_requirements.get("compute_score", 0)
                )
        
        analysis["estimated_time"] = total_time
        analysis["resource_requirements"] = total_resources
        
        return analysis
    
    def _classify_task_type(self, task: Task) -> str:
        """Classify task type based on query and context"""
        query_lower = task.query.lower()
        
        # Check for explicit task type
        if task.task_type != "general":
            return task.task_type
        
        # Pattern matching
        if any(keyword in query_lower for keyword in ["analyze code", "repository", "function", "class"]):
            return "code_analysis"
        
        if any(keyword in query_lower for keyword in ["research", "find information", "what is", "explain"]):
            return "research_query"
        
        if any(keyword in query_lower for keyword in ["document", "pdf", "file", "extract"]):
            return "document_processing"
        
        if "?" in task.query and len(task.query.split()) < 20:
            return "document_question"
        
        return "comprehensive_analysis"
    
    def _assess_complexity(self, task: Task) -> float:
        """Assess task complexity (0.0 to 1.0)"""
        complexity = 0.0
        
        # Query length factor
        word_count = len(task.query.split())
        if word_count > 50:
            complexity += 0.3
        elif word_count > 20:
            complexity += 0.2
        else:
            complexity += 0.1
        
        # Requirements complexity
        if len(task.requirements) > 3:
            complexity += 0.2
        elif len(task.requirements) > 1:
            complexity += 0.1
        
        # Context complexity
        if task.context and len(task.context) > 5:
            complexity += 0.2
        
        # Priority factor
        if task.priority == TaskPriority.URGENT:
            complexity += 0.1
        
        # Deadline pressure
        if task.deadline and task.deadline < datetime.now() + timedelta(hours=1):
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _infer_agents_from_query(self, query: str) -> List[AgentType]:
        """Infer required agents from query content"""
        agents = []
        query_lower = query.lower()
        
        # Always include RAG for general knowledge
        agents.append(AgentType.RAG)
        
        # Check for research indicators
        research_keywords = ["research", "latest", "recent", "current", "news", "trends", "compare"]
        if any(keyword in query_lower for keyword in research_keywords):
            agents.append(AgentType.RESEARCH)
        
        # Check for code analysis indicators
        code_keywords = ["code", "function", "class", "repository", "programming", "algorithm"]
        if any(keyword in query_lower for keyword in code_keywords):
            agents.append(AgentType.CODE_ANALYZER)
        
        # Check for document processing indicators
        doc_keywords = ["document", "pdf", "file", "extract", "parse", "analyze text"]
        if any(keyword in query_lower for keyword in doc_keywords):
            agents.append(AgentType.DOCUMENT_PROCESSOR)
        
        return agents

class ResponseSynthesizer:
    """Synthesizes responses from multiple agents"""
    
    def __init__(self, llm_handler: LLMHandler):
        self.llm_handler = llm_handler
    
    async def synthesize_responses(
        self, 
        task: Task, 
        agent_responses: List[AgentResponse],
        strategy: CoordinationStrategy
    ) -> Dict[str, Any]:
        """Synthesize multiple agent responses into coherent answer"""
        
        if not agent_responses:
            return {
                "synthesis": "No agent responses to synthesize.",
                "confidence": 0.0,
                "quality_metrics": {},
                "recommendations": []
            }
        
        # Filter successful responses
        successful_responses = [r for r in agent_responses if r.success]
        
        if not successful_responses:
            error_summary = self._create_error_summary(agent_responses)
            return {
                "synthesis": f"All agents encountered errors. {error_summary}",
                "confidence": 0.0,
                "quality_metrics": {"error_rate": 1.0},
                "recommendations": ["Retry with different parameters", "Check system status"]
            }
        
        # Create synthesis based on strategy
        if strategy == CoordinationStrategy.HIERARCHICAL:
            synthesis_result = await self._hierarchical_synthesis(task, successful_responses)
        elif len(successful_responses) == 1:
            synthesis_result = await self._single_agent_synthesis(task, successful_responses[0])
        else:
            synthesis_result = await self._multi_agent_synthesis(task, successful_responses)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(successful_responses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(task, successful_responses, quality_metrics)
        
        return {
            "synthesis": synthesis_result["text"],
            "confidence": synthesis_result["confidence"],
            "quality_metrics": quality_metrics,
            "recommendations": recommendations
        }
    
    async def _single_agent_synthesis(self, task: Task, response: AgentResponse) -> Dict[str, Any]:
        """Synthesize single agent response"""
        response_data = response.response_data
        
        if response.agent_type == AgentType.RAG:
            text = response_data.get("response", "No response available")
            confidence = response.confidence_score
        
        elif response.agent_type == AgentType.RESEARCH:
            summary = response_data.get("summary", "")
            key_findings = response_data.get("key_findings", [])
            
            text = f"{summary}\n\nKey Findings:\n"
            text += "\n".join([f"• {finding}" for finding in key_findings[:5]])
            confidence = response.confidence_score
        
        elif response.agent_type == AgentType.CODE_ANALYZER:
            insights = response_data.get("insights", {})
            recommendations = response_data.get("recommendations", [])
            
            text = "Code Analysis Results:\n"
            for category, data in insights.items():
                text += f"\n{category.title()}:\n"
                if isinstance(data, dict):
                    for key, value in data.items():
                        text += f"  {key}: {value}\n"
            
            if recommendations:
                text += "\nRecommendations:\n"
                text += "\n".join([f"• {rec}" for rec in recommendations])
            
            confidence = response.confidence_score
        
        elif response.agent_type == AgentType.DOCUMENT_PROCESSOR:
            metadata = response_data.get("metadata", {})
            chunks_count = response_data.get("total_chunks", 0)
            
            text = f"Document processed successfully.\n"
            text += f"Created {chunks_count} content chunks.\n"
            text += f"Document type: {metadata.get('file_type', 'unknown')}\n"
            
            confidence = response.confidence_score
        
        else:
            text = str(response_data)
            confidence = response.confidence_score
        
        return {"text": text, "confidence": confidence}
    
    async def _multi_agent_synthesis(self, task: Task, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Synthesize multiple agent responses using LLM"""
        
        # Prepare synthesis prompt
        synthesis_prompt = f"""
Task: {task.query}

Agent Responses:
"""
        
        response_texts = []
        total_confidence = 0.0
        
        for response in responses:
            agent_name = response.agent_type.value.upper()
            confidence = response.confidence_score
            total_confidence += confidence
            
            # Extract key information from each response
            if response.agent_type == AgentType.RAG:
                content = response.response_data.get("response", "")
                sources = response.response_data.get("sources", [])
                response_text = f"{agent_name} Agent (Confidence: {confidence:.1%}):\n{content}"
                if sources:
                    response_text += f"\nSources: {len(sources)} documents referenced"
            
            elif response.agent_type == AgentType.RESEARCH:
                summary = response.response_data.get("summary", "")
                key_findings = response.response_data.get("key_findings", [])
                source_count = response.response_data.get("source_count", 0)
                
                response_text = f"{agent_name} Agent (Confidence: {confidence:.1%}):\n{summary}"
                if key_findings:
                    response_text += f"\nKey findings: {', '.join(key_findings[:3])}"
                response_text += f"\nSources analyzed: {source_count}"
            
            elif response.agent_type == AgentType.CODE_ANALYZER:
                insights = response.response_data.get("insights", {})
                recommendations = response.response_data.get("recommendations", [])
                
                response_text = f"{agent_name} Agent (Confidence: {confidence:.1%}):\n"
                response_text += "Code analysis completed with insights on structure, complexity, and dependencies."
                if recommendations:
                    response_text += f"\nTop recommendations: {', '.join(recommendations[:2])}"
            
            elif response.agent_type == AgentType.DOCUMENT_PROCESSOR:
                total_chunks = response.response_data.get("total_chunks", 0)
                metadata = response.response_data.get("metadata", {})
                
                response_text = f"{agent_name} Agent (Confidence: {confidence:.1%}):\n"
                response_text += f"Document processed into {total_chunks} chunks."
                if metadata.get("file_type"):
                    response_text += f" File type: {metadata['file_type']}"
            
            else:
                response_text = f"{agent_name} Agent (Confidence: {confidence:.1%}):\n{str(response.response_data)}"
            
            response_texts.append(response_text)
        
        # Add all responses to prompt
        synthesis_prompt += "\n\n".join(response_texts)
        
        synthesis_prompt += f"""

Please provide a comprehensive synthesis that:
1. Integrates information from all agents
2. Addresses the original task directly
3. Highlights the most important insights
4. Resolves any conflicting information
5. Provides a clear, actionable conclusion

Synthesis:"""
        
        try:
            # Generate synthesis using LLM
            synthesized_text = await self.llm_handler.generate(
                prompt=synthesis_prompt,
                max_tokens=1024,
                temperature=0.3
            )
            
            # Calculate weighted confidence
            avg_confidence = total_confidence / len(responses) if responses else 0.0
            
            return {
                "text": synthesized_text.strip(),
                "confidence": avg_confidence
            }
            
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            
            # Fallback to simple concatenation
            fallback_text = f"Analysis of: {task.query}\n\n"
            fallback_text += "\n\n".join(response_texts)
            
            return {
                "text": fallback_text,
                "confidence": total_confidence / len(responses) if responses else 0.0
            }
    
    async def _hierarchical_synthesis(self, task: Task, responses: List[AgentResponse]) -> Dict[str, Any]:
        """Hierarchical synthesis with agent priority"""
        
        # Define agent hierarchy (higher priority agents influence synthesis more)
        agent_priority = {
            AgentType.RAG: 0.9,
            AgentType.RESEARCH: 0.8,
            AgentType.CODE_ANALYZER: 0.7,
            AgentType.DOCUMENT_PROCESSOR: 0.6
        }
        
        # Sort responses by priority
        sorted_responses = sorted(
            responses, 
            key=lambda r: agent_priority.get(r.agent_type, 0.5), 
            reverse=True
        )
        
        # Build synthesis starting with highest priority
        synthesis_parts = []
        primary_response = sorted_responses[0]
        
        # Start with primary agent response
        if primary_response.agent_type == AgentType.RAG:
            primary_text = primary_response.response_data.get("response", "")
            synthesis_parts.append(f"Primary Analysis: {primary_text}")
        
        # Add supporting information from other agents
        for response in sorted_responses[1:]:
            agent_name = response.agent_type.value.replace("_", " ").title()
            
            if response.agent_type == AgentType.RESEARCH:
                key_findings = response.response_data.get("key_findings", [])
                if key_findings:
                    synthesis_parts.append(f"{agent_name} Insights: {', '.join(key_findings[:3])}")
            
            elif response.agent_type == AgentType.CODE_ANALYZER:
                recommendations = response.response_data.get("recommendations", [])
                if recommendations:
                    synthesis_parts.append(f"{agent_name} Recommendations: {', '.join(recommendations[:2])}")
        
        # Calculate weighted confidence
        weighted_confidence = sum(
            response.confidence_score * agent_priority.get(response.agent_type, 0.5)
            for response in responses
        ) / sum(agent_priority.get(response.agent_type, 0.5) for response in responses)
        
        return {
            "text": "\n\n".join(synthesis_parts),
            "confidence": weighted_confidence
        }
    
    def _create_error_summary(self, agent_responses: List[AgentResponse]) -> str:
        """Create summary of agent errors"""
        error_agents = [r.agent_type.value for r in agent_responses if not r.success]
        return f"Failed agents: {', '.join(error_agents)}"
    
    def _calculate_quality_metrics(self, responses: List[AgentResponse]) -> Dict[str, float]:
        """Calculate quality metrics for responses"""
        if not responses:
            return {}
        
        total_responses = len(responses)
        successful_responses = len([r for r in responses if r.success])
        
        avg_confidence = sum(r.confidence_score for r in responses if r.success) / max(successful_responses, 1)
        avg_processing_time = sum(r.processing_time for r in responses) / total_responses
        
        return {
            "success_rate": successful_responses / total_responses,
            "average_confidence": avg_confidence,
            "average_processing_time": avg_processing_time,
            "response_consistency": self._calculate_consistency(responses)
        }
    
    def _calculate_consistency(self, responses: List[AgentResponse]) -> float:
        """Calculate consistency between agent responses"""
        # Simplified consistency measure
        if len(responses) < 2:
            return 1.0
        
        confidences = [r.confidence_score for r in responses if r.success]
        if not confidences:
            return 0.0
        
        # Measure variance in confidence scores
        if len(confidences) == 1:
            return 1.0
        
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        
        # Convert variance to consistency (0-1, where 1 is high consistency)
        return max(0.0, 1.0 - variance)
    
    def _generate_recommendations(
        self, 
        task: Task, 
        responses: List[AgentResponse], 
        quality_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Quality-based recommendations
        if quality_metrics.get("success_rate", 0) < 0.8:
            recommendations.append("Consider retrying with different parameters due to agent failures")
        
        if quality_metrics.get("average_confidence", 0) < 0.6:
            recommendations.append("Results have low confidence - consider additional verification")
        
        if quality_metrics.get("response_consistency", 0) < 0.7:
            recommendations.append("Agent responses show inconsistency - manual review recommended")
        
        # Agent-specific recommendations
        for response in responses:
            if response.agent_type == AgentType.RESEARCH and response.success:
                source_count = response.response_data.get("source_count", 0)
                if source_count < 3:
                    recommendations.append("Limited research sources found - consider broader search terms")
            
            elif response.agent_type == AgentType.CODE_ANALYZER and response.success:
                insights = response.response_data.get("insights", {})
                if insights.get("complexity", {}).get("high_complexity_functions", 0) > 5:
                    recommendations.append("High code complexity detected - consider refactoring")
        
        # Task-specific recommendations
        if task.priority == TaskPriority.URGENT and quality_metrics.get("average_processing_time", 0) > 30:
            recommendations.append("Processing time exceeded expectations for urgent task")
        
        return recommendations[:5]  # Limit to top 5 recommendations

class MultiAgentCoordinator:
    """Main coordinator class that orchestrates all agents"""
    
    def __init__(self):
        self.agent_manager = AgentManager()
        self.task_analyzer = TaskAnalyzer(self.agent_manager)
        self.response_synthesizer: Optional[ResponseSynthesizer] = None
        
        # Task management
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: Dict[str, CoordinationResult] = {}
        
        # Performance tracking
        self.coordination_stats = {
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_processing_time": 0.0,
            "strategy_usage": {strategy: 0 for strategy in CoordinationStrategy}
        }
        
        # Resource management
        self.max_concurrent_tasks = 5
        self.resource_limits = {
            "memory_mb": 4000,
            "compute_score": 1.0
        }
        
        # Background task executor
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def initialize(
        self,
        rag_orchestrator: RAGOrchestrator,
        document_processor: DocumentProcessor,
        code_analyzer: CodeAnalyzer,
        research_assistant: ResearchAssistant,
        llm_handler: LLMHandler
    ):
        """Initialize coordinator with all agents"""
        
        # Register agents
        self.agent_manager.register_agent(AgentType.RAG, rag_orchestrator)
        self.agent_manager.register_agent(AgentType.DOCUMENT_PROCESSOR, document_processor)
        self.agent_manager.register_agent(AgentType.CODE_ANALYZER, code_analyzer)
        self.agent_manager.register_agent(AgentType.RESEARCH, research_assistant)
        
        # Initialize response synthesizer
        self.response_synthesizer = ResponseSynthesizer(llm_handler)
        
        logger.info("Multi-agent coordinator initialized with all agents")
    
    async def coordinate_task(
        self,
        query: str,
        user_id: str,
        session_id: Optional[str] = None,
        task_type: str = "general",
        priority: TaskPriority = TaskPriority.NORMAL,
        context: Optional[Dict[str, Any]] = None,
        requirements: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        strategy_override: Optional[CoordinationStrategy] = None
    ) -> CoordinationResult:
        """Coordinate multiple agents to handle a complex task"""
        
        task_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        
        # Create task
        task = Task(
            task_id=task_id,
            query=query,
            user_id=user_id,
            session_id=session_id,
            task_type=task_type,
            priority=priority,
            context=context or {},
            requirements=requirements or [],
            constraints=constraints or {},
            created_at=datetime.now()
        )
        
        start_time = time.time()
        
        try:
            # Analyze task
            task_analysis = self.task_analyzer.analyze_task(task)
            
            # Override strategy if specified
            if strategy_override:
                task_analysis["coordination_strategy"] = strategy_override
            
            # Check resource availability
            if not self._check_resource_availability(task_analysis["resource_requirements"]):
                raise Exception("Insufficient resources available for task")
            
            # Add to active tasks
            self.active_tasks[task_id] = task
            
            # Execute coordination strategy
            strategy = task_analysis["coordination_strategy"]
            recommended_agents = task_analysis["recommended_agents"]
            
            logger.info(f"Executing {strategy.value} coordination with agents: {[a.value for a in recommended_agents]}")
            
            if strategy == CoordinationStrategy.PARALLEL:
                agent_responses = await self._execute_parallel_coordination(task, recommended_agents)
            elif strategy == CoordinationStrategy.SEQUENTIAL:
                agent_responses = await self._execute_sequential_coordination(task, recommended_agents)
            elif strategy == CoordinationStrategy.ADAPTIVE:
                agent_responses = await self._execute_adaptive_coordination(task, recommended_agents)
            elif strategy == CoordinationStrategy.HIERARCHICAL:
                agent_responses = await self._execute_hierarchical_coordination(task, recommended_agents)
            else:
                raise ValueError(f"Unknown coordination strategy: {strategy}")
            
            # Synthesize responses
            synthesis_result = await self.response_synthesizer.synthesize_responses(
                task, agent_responses, strategy
            )
            
            # Calculate overall metrics
            processing_time = time.time() - start_time
            overall_confidence = synthesis_result["confidence"]
            
            # Create result
            result = CoordinationResult(
                task_id=task_id,
                query=query,
                strategy_used=strategy,
                agents_involved=[r.agent_type for r in agent_responses if r.success],
                agent_responses=agent_responses,
                synthesized_response=synthesis_result["synthesis"],
                overall_confidence=overall_confidence,
                total_processing_time=processing_time,
                quality_metrics=synthesis_result["quality_metrics"],
                recommendations=synthesis_result["recommendations"],
                metadata={
                    "task_analysis": task_analysis,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Update statistics
            self._update_coordination_stats(result, True)
            
            # Store result
            self.completed_tasks[task_id] = result
            
            # Clean up
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            logger.info(f"Task {task_id} completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Task coordination failed: {e}")
            
            # Create error result
            error_result = CoordinationResult(
                task_id=task_id,
                query=query,
                strategy_used=CoordinationStrategy.PARALLEL,  # Default
                agents_involved=[],
                agent_responses=[],
                synthesized_response=f"Task coordination failed: {str(e)}",
                overall_confidence=0.0,
                total_processing_time=processing_time,
                quality_metrics={"error": True},
                recommendations=["Check system status", "Retry with simpler query"],
                metadata={"error": str(e), "session_id": session_id}
            )
            
            self._update_coordination_stats(error_result, False)
            
            # Clean up
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            return error_result
    
    async def _execute_parallel_coordination(self, task: Task, agents: List[AgentType]) -> List[AgentResponse]:
        """Execute agents in parallel"""
        agent_tasks = []
        
        for agent_type in agents:
            if self.agent_manager.is_agent_available(agent_type):
                agent_task = self._execute_single_agent(agent_type, task)
                agent_tasks.append(agent_task)
        
        if not agent_tasks:
            return []
        
        # Execute all agents in parallel
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Process results
        agent_responses = []
        for result in results:
            if isinstance(result, AgentResponse):
                agent_responses.append(result)
                self.agent_manager.update_agent_stats(result.agent_type, result)
            elif isinstance(result, Exception):
                logger.error(f"Agent execution failed: {result}")
        
        return agent_responses
    
    async def _execute_sequential_coordination(self, task: Task, agents: List[AgentType]) -> List[AgentResponse]:
        """Execute agents sequentially, passing context between them"""
        agent_responses = []
        accumulated_context = task.context.copy()
        
        for agent_type in agents:
            if self.agent_manager.is_agent_available(agent_type):
                # Update task context with previous results
                current_task = Task(
                    task_id=task.task_id,
                    query=task.query,
                    user_id=task.user_id,
                    session_id=task.session_id,
                    task_type=task.task_type,
                    priority=task.priority,
                    context=accumulated_context,
                    requirements=task.requirements,
                    constraints=task.constraints,
                    created_at=task.created_at,
                    metadata=task.metadata
                )
                
                response = await self._execute_single_agent(agent_type, current_task)
                agent_responses.append(response)
                self.agent_manager.update_agent_stats(agent_type, response)
                
                # Add response to context for next agent
                if response.success:
                    accumulated_context[f"{agent_type.value}_result"] = response.response_data
        
        return agent_responses
    
    async def _execute_adaptive_coordination(self, task: Task, agents: List[AgentType]) -> List[AgentResponse]:
        """Execute agents adaptively based on intermediate results"""
        agent_responses = []
        remaining_agents = agents.copy()
        
        # Start with highest priority agent
        if AgentType.RAG in remaining_agents:
            first_agent = AgentType.RAG
        else:
            first_agent = remaining_agents[0]
        
        # Execute first agent
        response = await self._execute_single_agent(first_agent, task)
        agent_responses.append(response)
        self.agent_manager.update_agent_stats(first_agent, response)
        remaining_agents.remove(first_agent)
        
        # Decide next agents based on first result
        if response.success and response.confidence_score > 0.8:
            # High confidence, execute only complementary agents
            complementary_agents = self._get_complementary_agents(first_agent, remaining_agents)
            for agent_type in complementary_agents[:2]:  # Limit to 2 additional
                agent_response = await self._execute_single_agent(agent_type, task)
                agent_responses.append(agent_response)
                self.agent_manager.update_agent_stats(agent_type, agent_response)
        
        elif response.success and response.confidence_score > 0.5:
            # Medium confidence, execute most relevant additional agent
            if remaining_agents:
                next_agent = self._select_best_agent(task, remaining_agents)
                agent_response = await self._execute_single_agent(next_agent, task)
                agent_responses.append(agent_response)
                self.agent_manager.update_agent_stats(next_agent, agent_response)
        
        else:
            # Low confidence or failure, try all remaining agents
            parallel_tasks = [
                self._execute_single_agent(agent_type, task) 
                for agent_type in remaining_agents[:3]  # Limit to 3
            ]
            
            if parallel_tasks:
                results = await asyncio.gather(*parallel_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, AgentResponse):
                        agent_responses.append(result)
                        self.agent_manager.update_agent_stats(result.agent_type, result)
        
        return agent_responses
    
    async def _execute_hierarchical_coordination(self, task: Task, agents: List[AgentType]) -> List[AgentResponse]:
        """Execute agents in hierarchical order based on task type"""
        # Define hierarchy based on task type
        task_hierarchies = {
            "research_query": [AgentType.RESEARCH, AgentType.RAG],
            "code_analysis": [AgentType.CODE_ANALYZER, AgentType.RAG],
            "document_processing": [AgentType.DOCUMENT_PROCESSOR, AgentType.RAG],
            "comprehensive_analysis": [AgentType.RAG, AgentType.RESEARCH, AgentType.CODE_ANALYZER]
        }
        
        task_type = self.task_analyzer._classify_task_type(task)
        hierarchy = task_hierarchies.get(task_type, [AgentType.RAG])
        
        # Filter available agents in hierarchy order
        ordered_agents = [agent for agent in hierarchy if agent in agents and self.agent_manager.is_agent_available(agent)]
        
        # Execute primary agent first
        agent_responses = []
        if ordered_agents:
            primary_response = await self._execute_single_agent(ordered_agents[0], task)
            agent_responses.append(primary_response)
            self.agent_manager.update_agent_stats(ordered_agents[0], primary_response)
            
            # Execute supporting agents in parallel if primary succeeded
            if primary_response.success and len(ordered_agents) > 1:
                supporting_tasks = [
                    self._execute_single_agent(agent_type, task) 
                    for agent_type in ordered_agents[1:]
                ]
                
                results = await asyncio.gather(*supporting_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, AgentResponse):
                        agent_responses.append(result)
                        self.agent_manager.update_agent_stats(result.agent_type, result)
        
        return agent_responses
    
    async def _execute_single_agent(self, agent_type: AgentType, task: Task) -> AgentResponse:
        """Execute a single agent"""
        start_time = time.time()
        
        try:
            agent = self.agent_manager.agents[agent_type]
            
            if agent_type == AgentType.RAG:
                result = await agent.process_query(
                    query=task.query,
                    user_id=task.user_id,
                    session_id=task.session_id,
                    max_tokens=512,
                    temperature=0.7
                )
                confidence = 0.8  # Default confidence for RAG
                
            elif agent_type == AgentType.RESEARCH:
                from tools.research_assistant import ResearchQuery
                research_query = ResearchQuery(
                    query=task.query,
                    user_id=task.user_id,
                    session_id=task.session_id,
                    query_type="factual"
                )
                result = await agent.research(research_query)
                confidence = result.confidence_score
                
            elif agent_type == AgentType.CODE_ANALYZER:
                # For code analysis, check if query is code-related
                if any(keyword in task.query.lower() for keyword in ['code', 'function', 'class', 'repository']):
                    search_results = await agent.search_code_elements(task.query, task.user_id, top_k=5)
                    result = {
                        "search_results": search_results,
                        "insights": {"structure": {"elements_found": len(search_results)}},
                        "recommendations": ["Review found code elements"] if search_results else ["No relevant code found"]
                    }
                    confidence = 0.9 if search_results else 0.3
                else:
                    result = {"message": "Query not related to code analysis"}
                    confidence = 0.1
                
            elif agent_type == AgentType.DOCUMENT_PROCESSOR:
                # For document processor, return capability info if no specific document
                result = {
                    "message": "Document processor ready for file uploads",
                    "supported_formats": ["pdf", "docx", "txt", "html", "md"],
                    "capabilities": ["text_extraction", "chunking", "metadata_analysis"]
                }
                confidence = 0.5
                
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            processing_time = time.time() - start_time
            
            return AgentResponse(
                agent_type=agent_type,
                task_id=task.task_id,
                response_data=result,
                confidence_score=confidence,
                processing_time=processing_time,
                success=True,
                metadata={"execution_time": processing_time}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Agent {agent_type.value} execution failed: {e}")
            
            return AgentResponse(
                agent_type=agent_type,
                task_id=task.task_id,
                response_data={},
                confidence_score=0.0,
                processing_time=processing_time,
                success=False,
                error_message=str(e),
                metadata={"execution_time": processing_time}
            )
    
    def _get_complementary_agents(self, primary_agent: AgentType, available_agents: List[AgentType]) -> List[AgentType]:
        """Get complementary agents for a primary agent"""
        complements = {
            AgentType.RAG: [AgentType.RESEARCH],
            AgentType.RESEARCH: [AgentType.RAG],
            AgentType.CODE_ANALYZER: [AgentType.RAG, AgentType.DOCUMENT_PROCESSOR],
            AgentType.DOCUMENT_PROCESSOR: [AgentType.RAG, AgentType.CODE_ANALYZER]
        }
        
        complementary = complements.get(primary_agent, [])
        return [agent for agent in complementary if agent in available_agents]
    
    def _select_best_agent(self, task: Task, available_agents: List[AgentType]) -> AgentType:
        """Select the best agent based on task characteristics and agent performance"""
        scores = {}
        
        for agent_type in available_agents:
            score = 0.0
            
            # Base capability score
            capability = self.agent_manager.get_agent_capability(agent_type)
            if capability:
                score += capability.quality_score
            
            # Performance history score
            stats = self.agent_manager.agent_stats[agent_type]
            if stats["total_tasks"] > 0:
                success_rate = stats["successful_tasks"] / stats["total_tasks"]
                score += success_rate * 0.5
                score += stats["avg_confidence"] * 0.3
            
            # Task relevance score
            query_lower = task.query.lower()
            if agent_type == AgentType.RESEARCH and any(keyword in query_lower for keyword in ["research", "find", "what is"]):
                score += 0.3
            elif agent_type == AgentType.CODE_ANALYZER and any(keyword in query_lower for keyword in ["code", "function", "algorithm"]):
                score += 0.3
            elif agent_type == AgentType.DOCUMENT_PROCESSOR and any(keyword in query_lower for keyword in ["document", "file", "extract"]):
                score += 0.3
            
            scores[agent_type] = score
        
        return max(scores, key=scores.get) if scores else available_agents[0]
    
    def _check_resource_availability(self, required_resources: Dict[str, float]) -> bool:
        """Check if required resources are available"""
        # Simplified resource check
        memory_required = required_resources.get("memory_mb", 0)
        compute_required = required_resources.get("compute_score", 0)
        
        return (memory_required <= self.resource_limits["memory_mb"] and 
                compute_required <= self.resource_limits["compute_score"])
    
    def _update_coordination_stats(self, result: CoordinationResult, success: bool):
        """Update coordination statistics"""
        self.coordination_stats["total_tasks"] += 1
        
        if success:
            self.coordination_stats["successful_tasks"] += 1
            
            # Update average processing time
            n = self.coordination_stats["successful_tasks"]
            current_avg = self.coordination_stats["avg_processing_time"]
            self.coordination_stats["avg_processing_time"] = (
                (current_avg * (n-1) + result.total_processing_time) / n
            )
        else:
            self.coordination_stats["failed_tasks"] += 1
        
        # Update strategy usage
        self.coordination_stats["strategy_usage"][result.strategy_used] += 1
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics"""
        return {
            **self.coordination_stats,
            "active_tasks_count": len(self.active_tasks),
            "completed_tasks_count": len(self.completed_tasks),
            "agent_stats": self.agent_manager.agent_stats,
            "agent_availability": {
                agent.value: self.agent_manager.is_agent_available(agent) 
                for agent in AgentType
            }
        }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "status": "active",
                "task": asdict(task),
                "started_at": task.created_at.isoformat()
            }
        elif task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                "status": "completed",
                "result": asdict(result)
            }
        else:
            return None
    
    async def cleanup(self):
        """Cleanup coordinator resources"""
        # Cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            logger.warning(f"Cancelling active task: {task_id}")
            del self.active_tasks[task_id]
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Multi-agent coordinator cleanup completed")

# Convenience functions for easy integration
async def create_coordinator(
    rag_orchestrator: RAGOrchestrator,
    document_processor: DocumentProcessor,
    code_analyzer: CodeAnalyzer,
    research_assistant: ResearchAssistant,
    llm_handler: LLMHandler
) -> MultiAgentCoordinator:
    """Create and initialize a multi-agent coordinator"""
    coordinator = MultiAgentCoordinator()
    await coordinator.initialize(
        rag_orchestrator,
        document_processor,
        code_analyzer,
        research_assistant,
        llm_handler
    )
    return coordinator

# Export main classes
__all__ = [
    'MultiAgentCoordinator',
    'AgentManager',
    'TaskAnalyzer',
    'ResponseSynthesizer',
    'Task',
    'AgentResponse',
    'CoordinationResult',
    'AgentType',
    'CoordinationStrategy',
    'TaskPriority',
    'create_coordinator'
]
"""
Multi-Agent Coordinator für mlx-langchain-lite
Orchestriert alle Enhanced Module für optimale Zusammenarbeit
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
import uuid
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import numpy as np

from tools.document_processor import DocumentProcessor, DocumentConfig
from tools.code_analyzer import CodeAnalyzer, CodeAnalysisConfig  
from tools.research_assistant import ResearchAssistant, ResearchConfig, ResearchQuery
from mlx_components.rag_orchestrator import RAGOrchestrator
from mlx_components.embedding_engine import EmbeddingEngine
from mlx_components.vector_store import VectorStore
from mlx_components.llm_handler import LLMHandler

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Available agent types"""
    RAG = "rag"
    RESEARCH = "research"
    CODE_ANALYZER = "code_analyzer"
    DOCUMENT_PROCESSOR = "document_processor"

class CoordinationStrategy(Enum):
    """Coordination strategies"""
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class AgentCapability:
    """Agent capability definition"""
    agent_type: AgentType
    capabilities: List[str]
    input_types: List[str]
    output_types: List[str]
    processing_time_estimate: float
    resource_requirements: Dict[str, float]
    quality_score: float = 0.0

@dataclass
class Task:
    """Task definition for multi-agent processing"""
    task_id: str
    query: str
    user_id: str
    session_id: str
    task_type: str
    priority: TaskPriority
    context: Dict[str, Any]
    requirements: List[str]
    constraints: Dict[str, Any]
    created_at: datetime
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class AgentResponse:
    """Response from an individual agent"""
    agent_type: AgentType
    task_id: str
    response_data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class CoordinationResult:
    """Final result from multi-agent coordination"""
    task_id: str
    query: str
    strategy_used: CoordinationStrategy
    agents_involved: List[AgentType]
    agent_responses: List[AgentResponse]
    synthesized_response: str
    overall_confidence: float
    total_processing_time: float
    quality_metrics: Dict[str, float]
    recommendations: List[str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class AgentManager:
    """Manages individual agents and their lifecycle"""
    
    def __init__(self):
        self.agents: Dict[AgentType, Any] = {}
        self.agent_capabilities: Dict[AgentType, AgentCapability] = {}
        self.agent_status: Dict[AgentType, str] = {}
        self.agent_stats: Dict[AgentType, Dict[str, Any]] = {}
        
        # Initialize agent capabilities
        self._initialize_capabilities()
    
    def _initialize_capabilities(self):
        """Initialize agent capabilities"""
        self.agent_capabilities = {
            AgentType.RAG: AgentCapability(
                agent_type=AgentType.RAG,
                capabilities=["document_retrieval", "semantic_search", "context_generation"],
                input_types=["text_query", "keywords"],
                output_types=["retrieved_documents", "context", "answers"],
                processing_time_estimate=2.0,
                resource_requirements={"memory_mb": 500, "compute_score": 0.7}
            ),
            
            AgentType.RESEARCH: AgentCapability(
                agent_type=AgentType.RESEARCH,
                capabilities=["web_search", "fact_checking", "source_validation", "trend_analysis"],
                input_types=["research_query", "topics", "keywords"],
                output_types=["research_report", "sources", "facts", "insights"],
                processing_time_estimate=15.0,
                resource_requirements={"memory_mb": 300, "compute_score": 0.5, "network": True}
            ),
            
            AgentType.CODE_