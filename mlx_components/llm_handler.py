"""
MLX LLM Handler
Integration mit mlx_parallm für High-Performance Batch LLM Processing
Optimiert für Gemma Models und Apple Silicon
"""

import asyncio
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

import mlx.core as mx
import mlx.nn as nn
from mlx_parallm.utils import load, batch_generate

@dataclass
class LLMConfig:
    """Konfiguration für LLM Handler"""
    model_path: str = "mlx-community/gemma-2-9b-it-4bit"
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9
    batch_size: int = 10
    max_batch_size: int = 50
    timeout_seconds: int = 120
    cache_responses: bool = True
    cache_size: int = 500
    format_prompts: bool = True

@dataclass
class ChatMessage:
    """Chat Message für Konversation"""
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class LLMRequest:
    """Request für LLM Processing"""
    prompt: str
    user_id: str
    conversation_id: Optional[str] = None
    context: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class LLMResponse:
    """Response von LLM Processing"""
    response: str
    prompt: str
    user_id: str
    model_name: str
    processing_time: float
    token_count: int
    cached: bool = False
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MLXLLMHandler:
    """
    High-Performance LLM Handler mit mlx_parallm Integration
    
    Features:
    - Batch Processing für maximale Effizienz
    - Multiple Gemma Model Support (2B, 9B, 27B)
    - Intelligent Response Caching
    - Context Assembly für RAG
    - Conversation Management
    - Performance Monitoring
    - Memory-efficient Model Loading
    """
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.model_cache = {}  # Multi-model support
        
        # Response Caching
        self.response_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance Metrics
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_processing_time = 0.0
        self.batch_count = 0
        
        # Conversation Management
        self.conversations = {}
        
        # Supported Models mit Performance Profilen
        self.supported_models = {
            "gemma-2b": {
                "path": "mlx-community/gemma-2-2b-it-4bit",
                "max_tokens": 2048,
                "optimal_batch_size": 20,
                "memory_usage": "Low"
            },
            "gemma-9b": {
                "path": "mlx-community/gemma-2-9b-it-4bit", 
                "max_tokens": 4096,
                "optimal_batch_size": 10,
                "memory_usage": "Medium"
            },
            "gemma-27b": {
                "path": "mlx-community/gemma-2-27b-it-4bit",
                "max_tokens": 8192,
                "optimal_batch_size": 5,
                "memory_usage": "High"
            }
        }
    
    async def initialize(self, model_path: Optional[str] = None) -> None:
        """
        Lazy Model Loading mit Caching für Multi-Model Support
        """
        target_model = model_path or self.config.model_path
        
        if target_model in self.model_cache:
            self.model, self.tokenizer = self.model_cache[target_model]
            self.model_name = target_model
            return
        
        if self.model is None or self.model_name != target_model:
            print(f"Loading LLM model: {target_model}")
            start_time = time.time()
            
            try:
                # Load model using mlx_parallm
                self.model, self.tokenizer = load(target_model)
                
                # Cache model for reuse
                self.model_cache[target_model] = (self.model, self.tokenizer)
                self.model_name = target_model
                
                load_time = time.time() - start_time
                print(f"✅ LLM Model loaded in {load_time:.2f}s")
                
            except Exception as e:
                print(f"❌ Error loading LLM model {target_model}: {e}")
                raise
    
    async def generate_single(self, 
                            request: LLMRequest, 
                            model_path: Optional[str] = None) -> LLMResponse:
        """
        Einzelne Response Generation
        """
        start_time = time.time()
        
        await self.initialize(model_path)
        
        # Check cache first
        cache_key = self._get_cache_key(request)
        if self.config.cache_responses and cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            self.cache_hits += 1
            
            return LLMResponse(
                response=cached_response["response"],
                prompt=request.prompt,
                user_id=request.user_id,
                model_name=self.model_name,
                processing_time=cached_response["processing_time"],
                token_count=cached_response["token_count"],
                cached=True,
                conversation_id=request.conversation_id,
                metadata=request.metadata
            )
        
        # Generate new response
        try:
            # Prepare prompt
            formatted_prompt = self._format_prompt(request)
            
            # Generate using mlx_parallm
            response_text = batch_generate(
                self.model,
                self.tokenizer,
                prompts=[formatted_prompt],
                max_tokens=request.max_tokens or self.config.max_tokens,
                temp=request.temperature or self.config.temperature,
                top_p=self.config.top_p,
                verbose=False,
                format_prompts=self.config.format_prompts
            )[0]
            
            # Clean response
            cleaned_response = self._clean_response(response_text, request.prompt)
            
            processing_time = time.time() - start_time
            token_count = self._estimate_token_count(cleaned_response)
            
            # Update cache
            if self.config.cache_responses:
                self._update_cache(cache_key, {
                    "response": cleaned_response,
                    "processing_time": processing_time,
                    "token_count": token_count
                })
            
            # Update metrics
            self.total_requests += 1
            self.total_tokens_generated += token_count
            self.total_processing_time += processing_time
            self.cache_misses += 1
            
            return LLMResponse(
                response=cleaned_response,
                prompt=request.prompt,
                user_id=request.user_id,
                model_name=self.model_name,
                processing_time=processing_time,
                token_count=token_count,
                cached=False,
                conversation_id=request.conversation_id,
                metadata=request.metadata
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return LLMResponse(
                response=f"Error generating response: {str(e)}",
                prompt=request.prompt,
                user_id=request.user_id,
                model_name=self.model_name,
                processing_time=time.time() - start_time,
                token_count=0,
                cached=False,
                conversation_id=request.conversation_id,
                metadata=request.metadata
            )
    
    async def generate_batch(self, 
                           requests: List[LLMRequest], 
                           model_path: Optional[str] = None) -> List[LLMResponse]:
        """
        Batch Response Generation für maximale Effizienz
        """
        start_time = time.time()
        
        await self.initialize(model_path)
        
        if not requests:
            return []
        
        # Check cache for existing responses
        cached_responses = {}
        non_cached_requests = []
        
        for i, request in enumerate(requests):
            cache_key = self._get_cache_key(request)
            if self.config.cache_responses and cache_key in self.response_cache:
                cached_responses[i] = self.response_cache[cache_key]
                self.cache_hits += 1
            else:
                non_cached_requests.append((i, request))
                self.cache_misses += 1
        
        # Process non-cached requests in batches
        new_responses = {}
        if non_cached_requests:
            # Split into optimal batch sizes
            batch_size = self._get_optimal_batch_size()
            
            for batch_start in range(0, len(non_cached_requests), batch_size):
                batch_end = min(batch_start + batch_size, len(non_cached_requests))
                batch_items = non_cached_requests[batch_start:batch_end]
                
                # Prepare batch prompts
                batch_prompts = [
                    self._format_prompt(request) 
                    for _, request in batch_items
                ]
                
                try:
                    # Batch generate using mlx_parallm
                    batch_responses = batch_generate(
                        self.model,
                        self.tokenizer,
                        prompts=batch_prompts,
                        max_tokens=self.config.max_tokens,
                        temp=self.config.temperature,
                        top_p=self.config.top_p,
                        verbose=False,
                        format_prompts=self.config.format_prompts
                    )
                    
                    # Process batch results
                    for j, (original_index, request) in enumerate(batch_items):
                        if j < len(batch_responses):
                            response_text = batch_responses[j]
                            cleaned_response = self._clean_response(response_text, request.prompt)
                            token_count = self._estimate_token_count(cleaned_response)
                            
                            response_data = {
                                "response": cleaned_response,
                                "processing_time": 0.0,  # Will be updated later
                                "token_count": token_count
                            }
                            
                            new_responses[original_index] = response_data
                            
                            # Update cache
                            if self.config.cache_responses:
                                cache_key = self._get_cache_key(request)
                                self._update_cache(cache_key, response_data)
                        else:
                            # Fallback for failed generations
                            new_responses[original_index] = {
                                "response": "Error: Generation failed",
                                "processing_time": 0.0,
                                "token_count": 0
                            }
                
                except Exception as e:
                    print(f"Error in batch generation: {e}")
                    # Fallback for entire batch
                    for original_index, request in batch_items:
                        new_responses[original_index] = {
                            "response": f"Error: {str(e)}",
                            "processing_time": 0.0,
                            "token_count": 0
                        }
        
        # Combine cached and new responses
        total_processing_time = time.time() - start_time
        results = []
        
        for i, request in enumerate(requests):
            if i in cached_responses:
                response_data = cached_responses[i]
                cached = True
            else:
                response_data = new_responses.get(i, {
                    "response": "Error: No response generated",
                    "processing_time": 0.0,
                    "token_count": 0
                })
                cached = False
            
            result = LLMResponse(
                response=response_data["response"],
                prompt=request.prompt,
                user_id=request.user_id,
                model_name=self.model_name,
                processing_time=total_processing_time / len(requests),
                token_count=response_data["token_count"],
                cached=cached,
                conversation_id=request.conversation_id,
                metadata=request.metadata
            )
            results.append(result)
        
        # Update metrics
        self.total_requests += len(requests)
        self.total_tokens_generated += sum(r.token_count for r in results)
        self.total_processing_time += total_processing_time
        self.batch_count += 1
        
        return results
    
    async def chat(self, 
                  messages: List[ChatMessage], 
                  user_id: str,
                  conversation_id: Optional[str] = None,
                  model_path: Optional[str] = None) -> LLMResponse:
        """
        Chat Interface mit Conversation Management
        """
        # Build conversation context
        conversation_prompt = self._build_conversation_prompt(messages)
        
        # Create request
        request = LLMRequest(
            prompt=conversation_prompt,
            user_id=user_id,
            conversation_id=conversation_id,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature
        )
        
        # Generate response
        response = await self.generate_single(request, model_path)
        
        # Update conversation history
        if conversation_id:
            self._update_conversation(conversation_id, messages, response.response)
        
        return response
    
    async def rag_generate(self, 
                          query: str, 
                          context_documents: List[Dict[str, Any]], 
                          user_id: str,
                          system_prompt: Optional[str] = None,
                          model_path: Optional[str] = None) -> LLMResponse:
        """
        RAG-optimierte Generation mit Context Assembly
        """
        # Build RAG prompt
        rag_prompt = self._build_rag_prompt(
            query=query,
            context_documents=context_documents,
            system_prompt=system_prompt
        )
        
        # Create request
        request = LLMRequest(
            prompt=rag_prompt,
            user_id=user_id,
            context=self._extract_context_summary(context_documents),
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            metadata={
                "query": query,
                "context_docs_count": len(context_documents),
                "type": "rag"
            }
        )
        
        return await self.generate_single(request, model_path)
    
    def _format_prompt(self, request: LLMRequest) -> str:
        """
        Formatiert Prompt für Gemma Models
        """
        if request.context:
            # RAG-style prompt with context
            return f"""<bos><start_of_turn>user
Context information:
{request.context}

Question: {request.prompt}

Please answer the question based on the provided context. If the context doesn't contain enough information, say so.
<end_of_turn>
<start_of_turn>model
"""
        else:
            # Standard prompt
            return f"""<bos><start_of_turn>user
{request.prompt}
<end_of_turn>
<start_of_turn>model
"""
    
    def _build_conversation_prompt(self, messages: List[ChatMessage]) -> str:
        """
        Baut Conversation Prompt aus Message History
        """
        prompt_parts = ["<bos>"]
        
        for message in messages:
            if message.role == "system":
                prompt_parts.append(f"<start_of_turn>system\n{message.content}<end_of_turn>")
            elif message.role == "user":
                prompt_parts.append(f"<start_of_turn>user\n{message.content}<end_of_turn>")
            elif message.role == "assistant":
                prompt_parts.append(f"<start_of_turn>model\n{message.content}<end_of_turn>")
        
        prompt_parts.append("<start_of_turn>model\n")
        
        return "\n".join(prompt_parts)
    
    def _build_rag_prompt(self, 
                         query: str, 
                         context_documents: List[Dict[str, Any]], 
                         system_prompt: Optional[str] = None) -> str:
        """
        Baut optimalen RAG Prompt
        """
        # Prepare context
        context_parts = []
        for i, doc in enumerate(context_documents[:5]):  # Limit to top 5 docs
            content = doc.get('content', doc.get('text', ''))
            metadata = doc.get('metadata', {})
            title = metadata.get('title', f'Document {i+1}')
            
            context_parts.append(f"## {title}\n{content}")
        
        context_text = "\n\n".join(context_parts)
        
        # Build prompt
        system_text = system_prompt or "You are a helpful AI assistant. Answer questions based on the provided context."
        
        return f"""<bos><start_of_turn>system
{system_text}
<end_of_turn>
<start_of_turn>user
Context Information:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above. Reference specific parts of the context when relevant.
<end_of_turn>
<start_of_turn>model
"""
    
    def _clean_response(self, response: str, original_prompt: str) -> str:
        """
        Reinigt Generated Response
        """
        # Remove prompt echoing
        if original_prompt in response:
            response = response.replace(original_prompt, "").strip()
        
        # Remove special tokens
        special_tokens = ["<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>", "model\n"]
        for token in special_tokens:
            response = response.replace(token, "")
        
        # Clean whitespace
        response = response.strip()
        
        # Remove empty lines at start
        lines = response.split('\n')
        while lines and not lines[0].strip():
            lines.pop(0)
        
        return '\n'.join(lines)
    
    def _get_optimal_batch_size(self) -> int:
        """
        Bestimmt optimale Batch-Größe basierend auf Model
        """
        for model_key, model_info in self.supported_models.items():
            if model_key in self.model_name:
                return min(model_info["optimal_batch_size"], self.config.max_batch_size)
        
        return self.config.batch_size
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Schätzt Token-Anzahl (grob: 1 Token ≈ 4 Zeichen)
        """
        return len(text) // 4
    
    def _get_cache_key(self, request: LLMRequest) -> str:
        """
        Erstellt Cache-Key für Request
        """
        content = f"{self.model_name}:{request.prompt}:{request.max_tokens}:{request.temperature}:{request.context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _update_cache(self, cache_key: str, response_data: Dict[str, Any]) -> None:
        """
        Updated Response Cache
        """
        # Cache size management
        if len(self.response_cache) >= self.config.cache_size:
            # Remove oldest entries
            oldest_keys = list(self.response_cache.keys())[:len(self.response_cache) - self.config.cache_size + 1]
            for key in oldest_keys:
                del self.response_cache[key]
        
        self.response_cache[cache_key] = response_data
    
    def _update_conversation(self, conversation_id: str, messages: List[ChatMessage], response: str) -> None:
        """
        Updated Conversation History
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        # Add new messages and response
        for message in messages:
            if message not in self.conversations[conversation_id]:
                self.conversations[conversation_id].append(message)
        
        # Add assistant response
        assistant_message = ChatMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat()
        )
        self.conversations[conversation_id].append(assistant_message)
        
        # Limit conversation length
        max_history = 20
        if len(self.conversations[conversation_id]) > max_history:
            self.conversations[conversation_id] = self.conversations[conversation_id][-max_history:]
    
    def _extract_context_summary(self, context_documents: List[Dict[str, Any]]) -> str:
        """
        Extrahiert Summary aus Context Documents
        """
        if not context_documents:
            return ""
        
        summaries = []
        for doc in context_documents[:3]:  # Top 3 for summary
            content = doc.get('content', doc.get('text', ''))
            # Take first 200 characters as summary
            summary = content[:200] + "..." if len(content) > 200 else content
            summaries.append(summary)
        
        return " | ".join(summaries)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Liefert Performance-Statistiken
        """
        avg_processing_time = self.total_processing_time / self.total_requests if self.total_requests > 0 else 0
        avg_tokens_per_request = self.total_tokens_generated / self.total_requests if self.total_requests > 0 else 0
        cache_hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        tokens_per_second = self.total_tokens_generated / self.total_processing_time if self.total_processing_time > 0 else 0
        
        return {
            "model_name": self.model_name,
            "total_requests": self.total_requests,
            "total_tokens_generated": self.total_tokens_generated,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_processing_time,
            "average_tokens_per_request": avg_tokens_per_request,
            "tokens_per_second": tokens_per_second,
            "batch_count": self.batch_count,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "active_conversations": len(self.conversations),
            "supported_models": list(self.supported_models.keys())
        }
    
    def clear_cache(self) -> None:
        """
        Leert Response Cache
        """
        self.response_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def clear_conversations(self) -> None:
        """
        Leert Conversation History
        """
        self.conversations.clear()
    
    async def benchmark(self) -> Dict[str, float]:
        """
        Performance Benchmark für LLM Handler
        """
        print("Running LLM Handler Benchmark...")
        
        # Test prompts
        test_prompts = [
            "What is machine learning?",
            "Explain neural networks in simple terms.",
            "How does natural language processing work?",
            "What are the benefits of using MLX for AI development?",
            "Describe the concept of transfer learning.",
        ] * 4  # 20 prompts total
        
        # Create test requests
        requests = [
            LLMRequest(
                prompt=prompt,
                user_id="benchmark_user",
                max_tokens=100,
                temperature=0.1
            ) for prompt in test_prompts
        ]
        
        # Clear cache for fair comparison
        self.clear_cache()
        
        # Benchmark single generation
        start_time = time.time()
        single_responses = []
        for request in requests[:5]:
            response = await self.generate_single(request)
            single_responses.append(response)
        single_time = time.time() - start_time
        
        # Benchmark batch generation
        start_time = time.time()
        batch_responses = await self.generate_batch(requests)
        batch_time = time.time() - start_time
        
        return {
            "single_requests": len(single_responses),
            "single_total_time": single_time,
            "single_requests_per_second": len(single_responses) / single_time,
            "batch_requests": len(batch_responses),
            "batch_total_time": batch_time,
            "batch_requests_per_second": len(batch_responses) / batch_time,
            "batch_speedup_factor": (single_time / len(single_responses)) / (batch_time / len(batch_responses)),
            "total_tokens_generated": sum(r.token_count for r in batch_responses),
            "tokens_per_second": sum(r.token_count for r in batch_responses) / batch_time
        }

# Usage Examples
async def example_usage():
    """Beispiele für LLM Handler Usage"""
    
    # Initialize with config
    config = LLMConfig(
        model_path="mlx-community/gemma-2-9b-it-4bit",
        batch_size=10,
        cache_responses=True
    )
    
    llm_handler = MLXLLMHandler(config)
    
    # Single generation
    request = LLMRequest(
        prompt="What are the advantages of using Apple Silicon for AI development?",
        user_id="user_123",
        max_tokens=200
    )
    
    response = await llm_handler.generate_single(request)
    print(f"Single Response: {response.response[:100]}...")
    print(f"Processing time: {response.processing_time:.3f}s")
    
    # Batch generation
    batch_requests = [
        LLMRequest(f"Explain {topic} in simple terms.", "user_123")
        for topic in ["machine learning", "neural networks", "transformers"]
    ]
    
    batch_responses = await llm_handler.generate_batch(batch_requests)
    print(f"Batch responses: {len(batch_responses)}")
    
    # Chat interface
    messages = [
        ChatMessage("system", "You are a helpful AI assistant."),
        ChatMessage("user", "What is MLX?"),
    ]
    
    chat_response = await llm_handler.chat(messages, "user_123", "conv_1")
    print(f"Chat response: {chat_response.response[:100]}...")
    
    # RAG generation
    context_docs = [
        {"content": "MLX is Apple's machine learning framework for Apple Silicon.", "metadata": {"title": "MLX Overview"}},
        {"content": "MLX provides high-performance ML operations on Apple Silicon.", "metadata": {"title": "MLX Performance"}}
    ]
    
    rag_response = await llm_handler.rag_generate(
        "What is MLX and why is it important?",
        context_docs,
        "user_123"
    )
    print(f"RAG response: {rag_response.response[:100]}...")
    
    # Performance stats
    stats = llm_handler.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    # Benchmark
    benchmark_results = await llm_handler.benchmark()
    print(f"Benchmark results: {benchmark_results}")

if __name__ == "__main__":
    asyncio.run(example_usage())