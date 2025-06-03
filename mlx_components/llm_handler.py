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
import logging # Added for robust logging

# Configure logging
# You can customize the logging level and format further if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# MLX and mlx_parallm imports
# These are assumed to be available in the execution environment.
# If mlx_parallm is not found, the code will rely on the user to handle it
# or it might raise an ImportError if not guarded appropriately by the user's setup.
try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_parallm.utils import load, batch_generate # Key dependency from the original code
    MLX_PARALLM_AVAILABLE = True
    logger.info("mlx.core, mlx.nn, and mlx_parallm.utils loaded successfully.")
except ImportError as e:
    MLX_PARALLM_AVAILABLE = False
    logger.warning(f"Failed to import mlx or mlx_parallm: {e}. Some functionalities might be limited or require mocks.")
    # Define placeholders if the library is not available, to allow the script to be parsed.
    # This is for development/linting purposes; actual execution would need the libraries.
    class MockModel: pass
    class MockTokenizer: pass
    def load(model_path: str): return MockModel(), MockTokenizer()
    def batch_generate(model, tokenizer, prompts, max_tokens, temp, top_p, verbose, format_prompts):
        logger.warning("Using mock 'batch_generate' due to import error.")
        return [f"Mock response for: {p[:30]}..." for p in prompts]

@dataclass
class LLMConfig:
    """Konfiguration für LLM Handler"""
    model_path: str = "mlx-community/gemma-2-9b-it-4bit"
    max_tokens: int = 1024
    temperature: float = 0.1
    top_p: float = 0.9 # Added for consistency with batch_generate parameters
    batch_size: int = 10 # Optimal batch size for self.model, used if mlx_parallm internal batching is not used
    max_batch_size: int = 50 # Max batch size mlx_parallm might handle or for splitting large requests
    timeout_seconds: int = 120 # Timeout for operations (conceptual, not directly used in current methods)
    cache_responses: bool = True
    cache_size: int = 500
    format_prompts: bool = True # Whether mlx_parallm should format prompts (Gemma specific)

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
    prompt: str # Can be a pre-formatted prompt or raw user input
    user_id: str
    conversation_id: Optional[str] = None
    context: Optional[str] = None # For RAG, context to be included in the prompt
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None # Added for more control
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class LLMResponse:
    """Response von LLM Processing"""
    response: str
    prompt: str # The actual prompt sent to the LLM
    user_id: str
    model_name: str
    processing_time: float
    token_count: int # Estimated tokens in the generated response
    cached: bool = False
    conversation_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MLXLLMHandler:
    """
    High-Performance LLM Handler mit mlx_parallm Integration.

    Features:
    - Batch Processing für maximale Effizienz via mlx_parallm.
    - Support für multiple Gemma Models (conceptual, relies on mlx_parallm loading).
    - Intelligent Response Caching.
    - Context Assembly für RAG.
    - Conversation Management.
    - Performance Monitoring.
    - Memory-efficient Model Loading (delegated to mlx_parallm's load function).
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        logger.info(f"Initializing MLXLLMHandler with config: {self.config}")

        self.model: Any = None # Stores the loaded MLX model
        self.tokenizer: Any = None # Stores the loaded tokenizer
        self.model_name: Optional[str] = None # Tracks the currently loaded model's path/name
        self.model_cache: Dict[str, Tuple[Any, Any]] = {}  # For caching multiple loaded models

        # Response Caching
        self.response_cache: Dict[str, Dict[str, Any]] = {} if self.config.cache_responses else None # type: ignore
        self.cache_hits = 0
        self.cache_misses = 0

        # Performance Metrics
        self.total_requests = 0
        self.total_tokens_generated = 0 # Based on _estimate_token_count
        self.total_processing_time = 0.0 # Sum of processing times for non-cached responses
        self.batch_count = 0 # Number of times generate_batch was called

        # Conversation Management
        self.conversations: Dict[str, List[ChatMessage]] = {}

        # Supported Models with Performance Profiles (conceptual, used for _get_optimal_batch_size)
        # The actual model loading and compatibility is handled by mlx_parallm.
        self.supported_models: Dict[str, Dict[str, Any]] = {
            "gemma-2b": {
                "path": "mlx-community/gemma-2-2b-it-4bit", # Example path
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
            # Add other models as needed, or make this dynamically configurable
        }
        logger.info(f"MLXLLMHandler initialized. Caching: {'Enabled' if self.config.cache_responses else 'Disabled'}")

    async def initialize(self, model_path: Optional[str] = None) -> None:
        """
        Lazy Model Loading mit Caching. Lädt ein Modell, wenn es noch nicht geladen ist
        oder wenn ein anderes Modell angefordert wird.
        Relies on `mlx_parallm.utils.load`.
        """
        target_model_path = model_path or self.config.model_path
        logger.debug(f"Initialize called for model path: {target_model_path}")

        if target_model_path in self.model_cache:
            self.model, self.tokenizer = self.model_cache[target_model_path]
            self.model_name = target_model_path
            logger.info(f"Model '{target_model_path}' loaded from cache.")
            return

        # If current model is different or no model loaded, then load
        if self.model is None or self.model_name != target_model_path:
            logger.info(f"Loading LLM model: {target_model_path}...")
            start_time = time.monotonic()
            try:
                if not MLX_PARALLM_AVAILABLE:
                    logger.error("mlx_parallm is not available. Cannot load model.")
                    raise ImportError("mlx_parallm.utils.load is required but not available.")

                # Load model using mlx_parallm's load function
                loaded_model, loaded_tokenizer = load(target_model_path)
                
                self.model = loaded_model
                self.tokenizer = loaded_tokenizer
                self.model_name = target_model_path
                
                # Cache the newly loaded model
                self.model_cache[target_model_path] = (self.model, self.tokenizer)
                
                load_time = time.monotonic() - start_time
                logger.info(f"✅ LLM Model '{target_model_path}' loaded in {load_time:.2f}s")

            except Exception as e:
                logger.error(f"❌ Error loading LLM model '{target_model_path}': {e}", exc_info=True)
                self.model = None # Ensure model state is clean on failure
                self.tokenizer = None
                self.model_name = None
                raise # Re-raise the exception to signal failure to the caller
        else:
            logger.info(f"Model '{target_model_path}' is already loaded and active.")


    async def generate_single(self,
                              request: LLMRequest,
                              model_path: Optional[str] = None) -> LLMResponse:
        """
        Generiert eine einzelne Antwort. Stellt sicher, dass das korrekte Modell initialisiert ist.
        """
        request_start_time = time.monotonic()
        logger.info(f"Generating single response for user '{request.user_id}', prompt snippet: '{request.prompt[:50]}...'")

        try:
            # Ensure the correct model is initialized (or the default one)
            await self.initialize(model_path or self.config.model_path)
            if not self.model or not self.tokenizer: # Check after initialize attempt
                 raise RuntimeError(f"Model '{self.model_name or self.config.model_path}' could not be loaded/initialized.")

        except Exception as init_error:
            logger.error(f"Initialization failed during generate_single: {init_error}", exc_info=True)
            return LLMResponse(
                response=f"Error: Model initialization failed: {str(init_error)}",
                prompt=request.prompt, user_id=request.user_id, model_name=str(model_path or self.config.model_path),
                processing_time=time.monotonic() - request_start_time, token_count=0, cached=False,
                conversation_id=request.conversation_id, metadata=request.metadata
            )

        # Check cache first
        cache_key = self._get_cache_key(request)
        if self.config.cache_responses and self.response_cache is not None and cache_key in self.response_cache:
            cached_data = self.response_cache[cache_key]
            self.cache_hits += 1
            processing_time_for_cached = time.monotonic() - request_start_time # Time to check cache
            logger.info(f"Cache hit for user '{request.user_id}'. Key: {cache_key}")
            return LLMResponse(
                response=cached_data["response"],
                prompt=request.prompt, # Original request prompt
                user_id=request.user_id,
                model_name=str(self.model_name),
                processing_time=processing_time_for_cached, # Use actual time for cache hit
                token_count=cached_data["token_count"],
                cached=True,
                conversation_id=request.conversation_id,
                metadata=request.metadata
            )

        # Generate new response
        generation_start_time = time.monotonic()
        try:
            formatted_prompt = self._format_prompt(request) # Format based on request type (RAG, chat, etc.)
            
            # Use mlx_parallm's batch_generate even for a single prompt, as it might be optimized
            # Ensure mlx_parallm is available
            if not MLX_PARALLM_AVAILABLE:
                raise ImportError("mlx_parallm.utils.batch_generate is required but not available.")

            # Note: batch_generate is synchronous in the provided snippet.
            # If it's truly blocking, for an async handler, it should be run in an executor.
            # loop = asyncio.get_event_loop()
            # response_texts = await loop.run_in_executor(None, batch_generate, ...)
            # For now, calling it directly as in the original code.
            response_texts = batch_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompts=[formatted_prompt],
                max_tokens=request.max_tokens or self.config.max_tokens,
                temp=request.temperature or self.config.temperature,
                top_p=request.top_p or self.config.top_p,
                verbose=False, # Typically false for programmatic use
                format_prompts=self.config.format_prompts # Let mlx_parallm handle Gemma formatting if true
            )
            response_text = response_texts[0] if response_texts else ""
            
            cleaned_response = self._clean_response(response_text, formatted_prompt) # Clean based on formatted prompt
            
            current_processing_time = time.monotonic() - generation_start_time
            estimated_token_count = self._estimate_token_count(cleaned_response)

            # Update cache
            if self.config.cache_responses and self.response_cache is not None:
                self._update_cache(cache_key, {
                    "response": cleaned_response,
                    "processing_time": current_processing_time, # Cache processing time of generation itself
                    "token_count": estimated_token_count
                })

            # Update metrics for non-cached responses
            self.total_requests += 1
            self.total_tokens_generated += estimated_token_count
            self.total_processing_time += current_processing_time
            self.cache_misses += 1
            
            logger.info(f"Generated new response for user '{request.user_id}' in {current_processing_time:.3f}s.")
            return LLMResponse(
                response=cleaned_response,
                prompt=formatted_prompt, # Return the actual prompt sent to LLM
                user_id=request.user_id,
                model_name=str(self.model_name),
                processing_time=time.monotonic() - request_start_time, # Total time for this request
                token_count=estimated_token_count,
                cached=False,
                conversation_id=request.conversation_id,
                metadata=request.metadata
            )

        except Exception as e:
            logger.error(f"Error generating single response: {e}", exc_info=True)
            return LLMResponse(
                response=f"Error generating response: {str(e)}",
                prompt=request.prompt, # Original request prompt
                user_id=request.user_id,
                model_name=str(self.model_name),
                processing_time=time.monotonic() - request_start_time,
                token_count=0,
                cached=False,
                conversation_id=request.conversation_id,
                metadata=request.metadata
            )

    async def generate_batch(self,
                             requests: List[LLMRequest],
                             model_path: Optional[str] = None) -> List[LLMResponse]:
        """
        Batch Response Generation für maximale Effizienz.
        Handles caching for individual requests within the batch.
        """
        batch_start_time = time.monotonic()
        logger.info(f"Starting batch generation for {len(requests)} requests.")

        if not requests:
            return []

        try:
            await self.initialize(model_path or self.config.model_path)
            if not self.model or not self.tokenizer:
                 raise RuntimeError(f"Model '{self.model_name or self.config.model_path}' could not be loaded/initialized for batch.")
        except Exception as init_error:
            logger.error(f"Initialization failed during generate_batch: {init_error}", exc_info=True)
            error_response_list: List[LLMResponse] = []
            for req in requests:
                error_response_list.append(LLMResponse(
                    response=f"Error: Model initialization failed for batch: {str(init_error)}",
                    prompt=req.prompt, user_id=req.user_id, model_name=str(model_path or self.config.model_path),
                    processing_time=time.monotonic() - batch_start_time, token_count=0, cached=False,
                    conversation_id=req.conversation_id, metadata=req.metadata
                ))
            return error_response_list


        final_llm_responses: List[Optional[LLMResponse]] = [None] * len(requests)
        non_cached_indices_requests: List[Tuple[int, LLMRequest]] = []

        # Check cache for each request
        if self.config.cache_responses and self.response_cache is not None:
            for i, request_obj in enumerate(requests): # Renamed to avoid conflict
                cache_key = self._get_cache_key(request_obj)
                if cache_key in self.response_cache:
                    cached_data = self.response_cache[cache_key]
                    self.cache_hits += 1
                    logger.debug(f"Batch cache hit for request index {i}, key: {cache_key}")
                    final_llm_responses[i] = LLMResponse(
                        response=cached_data["response"], prompt=request_obj.prompt, user_id=request_obj.user_id,
                        model_name=str(self.model_name), processing_time=cached_data["processing_time"], # Use cached processing time
                        token_count=cached_data["token_count"], cached=True,
                        conversation_id=request_obj.conversation_id, metadata=request_obj.metadata
                    )
                else:
                    non_cached_indices_requests.append((i, request_obj))
                    self.cache_misses += 1 # For overall stats
        else: # Caching disabled, all requests are non-cached
            non_cached_indices_requests = list(enumerate(requests))


        # Process non-cached requests in batches
        if non_cached_indices_requests:
            logger.info(f"Processing {len(non_cached_indices_requests)} non-cached requests in batch.")
            # Determine optimal batch size for the current model
            # This is conceptual if mlx_parallm handles its own batching internally.
            # The config.batch_size might be a hint for mlx_parallm or for splitting here.
            processing_batch_size = self._get_optimal_batch_size()

            for i in range(0, len(non_cached_indices_requests), processing_batch_size):
                current_batch_slice = non_cached_indices_requests[i : i + processing_batch_size]
                original_indices_in_slice = [item[0] for item in current_batch_slice]
                requests_in_slice = [item[1] for item in current_batch_slice]

                batch_prompts_to_generate = [self._format_prompt(req) for req in requests_in_slice]
                
                generation_start_time_slice = time.monotonic()
                try:
                    if not MLX_PARALLM_AVAILABLE:
                        raise ImportError("mlx_parallm.utils.batch_generate is required but not available for batch processing.")

                    # Actual call to mlx_parallm's batch_generate
                    generated_texts = batch_generate(
                        model=self.model, tokenizer=self.tokenizer,
                        prompts=batch_prompts_to_generate,
                        max_tokens=requests_in_slice[0].max_tokens or self.config.max_tokens, # Assuming same max_tokens for batch
                        temp=requests_in_slice[0].temperature or self.config.temperature, # Assuming same temp
                        top_p=requests_in_slice[0].top_p or self.config.top_p, # Assuming same top_p
                        verbose=False, format_prompts=self.config.format_prompts
                    )
                    slice_processing_time = time.monotonic() - generation_start_time_slice
                    avg_time_per_item_in_slice = slice_processing_time / len(requests_in_slice) if requests_in_slice else 0

                    for idx_in_slice, original_req_idx in enumerate(original_indices_in_slice):
                        original_request_obj = requests[original_req_idx]
                        if idx_in_slice < len(generated_texts):
                            raw_response_text = generated_texts[idx_in_slice]
                            cleaned_response = self._clean_response(raw_response_text, batch_prompts_to_generate[idx_in_slice])
                            token_count = self._estimate_token_count(cleaned_response)

                            response_data_for_cache = {
                                "response": cleaned_response,
                                "processing_time": avg_time_per_item_in_slice, # Approximate time for this item
                                "token_count": token_count
                            }
                            if self.config.cache_responses and self.response_cache is not None:
                                self._update_cache(self._get_cache_key(original_request_obj), response_data_for_cache)

                            final_llm_responses[original_req_idx] = LLMResponse(
                                response=cleaned_response, prompt=batch_prompts_to_generate[idx_in_slice],
                                user_id=original_request_obj.user_id, model_name=str(self.model_name),
                                processing_time=avg_time_per_item_in_slice, token_count=token_count, cached=False,
                                conversation_id=original_request_obj.conversation_id, metadata=original_request_obj.metadata
                            )
                            self.total_tokens_generated += token_count
                        else: # Should not happen if batch_generate returns one for each prompt
                            logger.warning(f"Missing response for request index {original_req_idx} in batch results.")
                            final_llm_responses[original_req_idx] = LLMResponse(
                                response="Error: No response generated in batch", prompt=batch_prompts_to_generate[idx_in_slice],
                                user_id=original_request_obj.user_id, model_name=str(self.model_name),
                                processing_time=avg_time_per_item_in_slice, token_count=0, cached=False,
                                conversation_id=original_request_obj.conversation_id, metadata=original_request_obj.metadata
                            )
                    self.total_requests += len(requests_in_slice) # Count these as processed requests
                    self.total_processing_time += slice_processing_time


                except Exception as e:
                    logger.error(f"Error during batch generation for slice starting at index {original_indices_in_slice[0]}: {e}", exc_info=True)
                    slice_fail_time = time.monotonic() - generation_start_time_slice
                    avg_fail_time = slice_fail_time / len(requests_in_slice) if requests_in_slice else 0
                    for original_req_idx in original_indices_in_slice:
                        original_request_obj = requests[original_req_idx]
                        final_llm_responses[original_req_idx] = LLMResponse(
                            response=f"Error during batch processing: {str(e)}", prompt=self._format_prompt(original_request_obj),
                            user_id=original_request_obj.user_id, model_name=str(self.model_name),
                            processing_time=avg_fail_time, token_count=0, cached=False,
                            conversation_id=original_request_obj.conversation_id, metadata=original_request_obj.metadata
                        )
            self.batch_count +=1 # Increment for the overall batch call, not sub-batches.

        # Fill in any Nones that might have occurred if logic missed a case (should not happen)
        for i in range(len(requests)):
            if final_llm_responses[i] is None:
                req = requests[i]
                logger.error(f"Critical error: Response for request index {i} was not set in batch processing.")
                final_llm_responses[i] = LLMResponse(
                    response="Error: Processing failed unexpectedly.", prompt=req.prompt, user_id=req.user_id,
                    model_name=str(self.model_name), processing_time=0, token_count=0, cached=False,
                    conversation_id=req.conversation_id, metadata=req.metadata
                )
        
        total_batch_processing_time = time.monotonic() - batch_start_time
        logger.info(f"Batch generation completed for {len(requests)} requests in {total_batch_processing_time:.3f}s.")
        # Adjust individual processing times for cached items to reflect total batch time for user perspective
        for i, resp in enumerate(final_llm_responses):
            if resp: # resp should not be None here
                 # This is tricky. The cached item's 'processing_time' is its original generation time.
                 # The overall time for this batch call is total_batch_processing_time.
                 # For simplicity, we'll leave cached item processing_time as is, and new items have their slice-avg.
                 # The user of this batch method should look at the overall time for the batch.
                 pass


        return final_llm_responses # type: ignore

    async def chat(self,
                   messages: List[ChatMessage],
                   user_id: str,
                   conversation_id: Optional[str] = None,
                   model_path: Optional[str] = None,
                   max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None) -> LLMResponse:
        """
        Chat Interface mit Conversation Management.
        Builds a conversation prompt and calls generate_single.
        """
        logger.info(f"Processing chat for user '{user_id}', conversation ID: '{conversation_id}'. Messages count: {len(messages)}")
        if not messages:
            logger.warning("Chat called with no messages.")
            # Or return an error/empty response
            return LLMResponse(response="Error: No messages provided for chat.", prompt="", user_id=user_id, model_name="", processing_time=0, token_count=0)


        # Build conversation prompt from message history
        conversation_prompt_str = self._build_conversation_prompt(messages)

        # Create LLMRequest
        llm_req = LLMRequest(
            prompt=conversation_prompt_str, # The fully constructed chat prompt
            user_id=user_id,
            conversation_id=conversation_id,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            metadata={"type": "chat", "message_count": len(messages)}
        )

        # Generate response using the single generation logic
        llm_resp = await self.generate_single(llm_req, model_path)

        # Update conversation history if a conversation_id is provided and response was successful
        if conversation_id and not llm_resp.response.startswith("Error:"):
            # The original messages are already part of the input `messages` list.
            # We only need to add the assistant's response.
            self._update_conversation(conversation_id, messages, llm_resp.response)
        
        # The llm_resp.prompt will be the full conversation_prompt_str
        return llm_resp

    async def rag_generate(self,
                           query: str,
                           context_documents: List[Dict[str, Any]],
                           user_id: str,
                           system_prompt: Optional[str] = None,
                           model_path: Optional[str] = None,
                           max_tokens: Optional[int] = None,
                           temperature: Optional[float] = None) -> LLMResponse:
        """
        RAG-optimierte Generation mit Context Assembly.
        Builds a RAG-specific prompt and calls generate_single.
        """
        logger.info(f"Processing RAG generate for user '{user_id}', query: '{query[:50]}...'. Context docs: {len(context_documents)}")

        # Build RAG prompt
        rag_prompt_str = self._build_rag_prompt(
            query=query,
            context_documents=context_documents,
            system_prompt=system_prompt
        )

        # Create LLMRequest
        llm_req = LLMRequest(
            prompt=rag_prompt_str, # The fully constructed RAG prompt
            user_id=user_id,
            context=self._extract_context_summary(context_documents), # Summary for potential use in request logging/metadata
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            metadata={
                "original_query": query, # Keep original query for clarity
                "context_docs_count": len(context_documents),
                "type": "rag"
            }
        )

        return await self.generate_single(llm_req, model_path)

    def _format_prompt(self, request: LLMRequest) -> str:
        """
        Formatiert Prompt für Gemma Models oder andere Modelle basierend auf request type.
        This method is crucial for model-specific instruction formatting.
        The current implementation is Gemma-centric.
        """
        # If request.prompt is already a fully formatted model-specific prompt (e.g. from chat or RAG builders),
        # it might be used directly. Otherwise, construct it.
        # The current logic in generate_single, chat, rag_generate already prepares specific prompts.
        # This method could be a final wrapper if needed, or primarily used by generate_single if request.prompt is raw.

        # If the prompt is coming from chat() or rag_generate(), it's already formatted.
        # If it's a direct call to generate_single() with a raw prompt, then format it.
        
        # Check if the prompt seems to be already formatted (heuristic)
        if "<start_of_turn>" in request.prompt and "<end_of_turn>" in request.prompt:
            logger.debug("Prompt appears to be pre-formatted, using as-is.")
            return request.prompt # Assume it's already correctly formatted

        # Default formatting for a raw user prompt (Gemma style)
        # This is similar to what _build_conversation_prompt would do for a single user message.
        logger.debug("Applying default Gemma formatting to raw prompt.")
        formatted = f"<bos><start_of_turn>user\n{request.prompt}<end_of_turn>\n<start_of_turn>model\n"
        
        # If RAG-style context is provided in LLMRequest, incorporate it
        if request.context: # This context is a summary, not the full RAG context for the prompt
            # This part is a bit ambiguous. _build_rag_prompt is more specific.
            # If request.context is meant to be part of a generic prompt:
            logger.debug("Incorporating request.context into formatted prompt.")
            # This might conflict with _build_rag_prompt. Prioritize specific builders.
            # For a generic prompt with context:
            generic_context_prompt = f"""<bos><start_of_turn>user
Context: {request.context}

Question: {request.prompt}
<end_of_turn>
<start_of_turn>model
"""
            # Decide which formatting to use if request.context is present.
            # For now, let's assume if request.prompt is not pre-formatted, and context exists,
            # it's a simple Q&A with context.
            return generic_context_prompt


        return formatted


    def _build_conversation_prompt(self, messages: List[ChatMessage]) -> str:
        """Baut Conversation Prompt aus Message History für Gemma-artige Modelle."""
        prompt_parts = ["<bos>"] # Beginning of sequence

        for message in messages:
            if message.role == "system":
                # System prompts are often at the beginning, outside user/model turns,
                # or as the first part of the first user turn for some models.
                # For Gemma, it's usually: <start_of_turn>user\nSYSTEM_PROMPT\nUSER_QUERY<end_of_turn>
                # Or, if truly a system message before conversation:
                prompt_parts.append(f"<start_of_turn>system\n{message.content}<end_of_turn>") # Check model docs
            elif message.role == "user":
                prompt_parts.append(f"<start_of_turn>user\n{message.content}<end_of_turn>")
            elif message.role == "assistant":
                prompt_parts.append(f"<start_of_turn>model\n{message.content}<end_of_turn>")
            else:
                logger.warning(f"Unknown message role: {message.role}. Skipping message.")

        # Add the final turn to signal the model to generate
        prompt_parts.append("<start_of_turn>model\n") # Model's turn to speak

        return "\n".join(prompt_parts)

    def _build_rag_prompt(self,
                          query: str,
                          context_documents: List[Dict[str, Any]],
                          system_prompt: Optional[str] = None) -> str:
        """Baut optimalen RAG Prompt für Gemma-artige Modelle."""
        context_parts = []
        # Limit number of documents and total context length to avoid exceeding model limits
        # This should be coordinated with RAGConfig.max_context_length if that's char-based
        # For now, simple document count limit.
        for i, doc in enumerate(context_documents[:3]):  # Example: Top 3-5 documents
            content = doc.get('content', doc.get('text', '')) # Prefer 'content', fallback to 'text'
            metadata = doc.get('metadata', {})
            title = metadata.get('title', metadata.get('filename', f'Source {i+1}'))

            # Truncate individual document content if very long
            max_doc_len = 1500 # Characters per document in prompt
            truncated_content = content[:max_doc_len] + "..." if len(content) > max_doc_len else content
            context_parts.append(f"Source {i+1}: {title}\nContent:\n{truncated_content}")

        context_text = "\n\n---\n\n".join(context_parts)
        if not context_text:
            context_text = "No context documents were provided."

        system_message = system_prompt or \
            "You are a helpful AI assistant. Answer the user's question based on the provided sources. " \
            "If the answer is not found in the sources, state that clearly. " \
            "Cite the sources used in your answer (e.g., 'According to Source 1: ...')."

        # Gemma-specific RAG prompt structure
        # Reference: Google's Gemma documentation for instruction-tuned models.
        # The format is typically:
        # <start_of_turn>user
        # [Instruction/System Prompt (optional)]
        # [Context Documents]
        # [Question]
        # <end_of_turn>
        # <start_of_turn>model
        # (Model generates here)

        prompt = f"""<bos><start_of_turn>user
{system_message}

Here are some documents that might be relevant:
{context_text}

Based on these documents, please answer the following question:
Question: {query}
<end_of_turn>
<start_of_turn>model
"""
        return prompt

    def _clean_response(self, response: str, original_prompt_sent_to_llm: str) -> str:
        """
        Reinigt die generierte Antwort des LLMs.
        Removes echoed prompts and special tokens.
        """
        logger.debug(f"Raw response before cleaning: '{response[:200]}...'")
        
        # More robust prompt echoing removal:
        # Sometimes the model might slightly rephrase the tail of the prompt.
        # We look for the start of the model's actual generation.
        # For Gemma, this is often right after "<start_of_turn>model\n"
        model_turn_signal = "<start_of_turn>model\n"
        if model_turn_signal in original_prompt_sent_to_llm:
            # If the prompt itself contained this signal (e.g. it was passed pre-formatted)
            # we need to be careful. The actual response starts after the *last* such signal in the prompt.
            last_signal_idx_in_prompt = original_prompt_sent_to_llm.rfind(model_turn_signal)
            # The response should start after this signal in the combined output
            if response.startswith(original_prompt_sent_to_llm[:last_signal_idx_in_prompt + len(model_turn_signal)]):
                 response = response[last_signal_idx_in_prompt + len(model_turn_signal):]


        # Fallback: if the exact prompt is echoed
        # This can be problematic if the prompt is part of the desired answer.
        # if response.startswith(original_prompt_sent_to_llm):
        #    response = response[len(original_prompt_sent_to_llm):]

        # Remove common special tokens that might remain
        # Gemma specific tokens: <bos>, <eos> (often handled by generation params)
        # <start_of_turn>, <end_of_turn> might appear if not stripped by tokenizer/generation
        special_tokens_to_remove = ["<bos>", "<eos>", "<pad>"] # <pad> if applicable
        # Model turn signals should ideally not be in the final output if generation stops correctly.
        # However, sometimes they might leak.
        # "<start_of_turn>user", "<start_of_turn>model", "<end_of_turn>"
        
        for token in special_tokens_to_remove:
            response = response.replace(token, "")

        # Strip leading/trailing whitespace
        cleaned = response.strip()
        
        # Remove any leading "model\n" or "assistant\n" if it wasn't part of a structured turn
        if cleaned.startswith("model\n"):
            cleaned = cleaned[len("model\n"):].lstrip()
        elif cleaned.startswith("assistant\n"):
            cleaned = cleaned[len("assistant\n"):].lstrip()

        logger.debug(f"Cleaned response: '{cleaned[:200]}...'")
        return cleaned


    def _get_optimal_batch_size(self) -> int:
        """Bestimmt die optimale Batch-Größe basierend auf dem aktuell geladenen Modell."""
        if not self.model_name:
            logger.warning("No model loaded, using default batch size from config.")
            return self.config.batch_size

        for model_key_prefix, model_info in self.supported_models.items():
            # Check if the loaded model_name contains the key prefix (e.g., "gemma-9b" in "mlx-community/gemma-2-9b-it-4bit")
            if model_key_prefix in self.model_name:
                optimal_size = model_info.get("optimal_batch_size", self.config.batch_size)
                # Ensure it doesn't exceed the handler's configured max_batch_size
                return min(optimal_size, self.config.max_batch_size)
        
        logger.info(f"No specific profile for '{self.model_name}', using default batch size: {self.config.batch_size}")
        return self.config.batch_size

    def _estimate_token_count(self, text: str) -> int:
        """Schätzt die Token-Anzahl. Eine grobe Schätzung, für genaue Zählung ist der Tokenizer nötig."""
        # A common heuristic: 1 token ~ 4 characters in English.
        # This is very approximate. For more accuracy, use tokenizer.encode(text)
        # if self.tokenizer:
        # try:
        # return len(self.tokenizer.encode(text))
        # except Exception: # Fallback if tokenizer or encode method is problematic
        # pass
        return max(1, len(text) // 4) # Ensure at least 1 for non-empty strings

    def _get_cache_key(self, request: LLMRequest) -> str:
        """Erstellt einen Cache-Key für einen LLMRequest."""
        # Include all parameters that affect the response
        key_components = [
            self.model_name or self.config.model_path, # Current model is part of the key
            request.prompt, # The core input
            str(request.max_tokens or self.config.max_tokens),
            str(request.temperature or self.config.temperature),
            str(request.top_p or self.config.top_p),
            str(request.context or ""), # RAG context if any
            # Potentially add other relevant fields from request or config if they vary and affect output
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode('utf-8')).hexdigest()

    def _update_cache(self, cache_key: str, response_data: Dict[str, Any]) -> None:
        """Updated den Response Cache und verwaltet dessen Größe."""
        if self.response_cache is None: # Cache disabled
            return

        if len(self.response_cache) >= self.config.cache_size:
            # FIFO eviction: remove the oldest entry.
            # Python dicts maintain insertion order from 3.7+.
            try:
                oldest_key = next(iter(self.response_cache)) # Get the first key (oldest)
                del self.response_cache[oldest_key]
                logger.debug(f"Cache full. Evicted oldest entry: {oldest_key}")
            except StopIteration: # Should not happen if cache_size >= 1 and cache is full
                logger.warning("Cache eviction failed: cache reported full but was empty.")
        
        self.response_cache[cache_key] = response_data
        logger.debug(f"Cached response for key: {cache_key}. Cache size: {len(self.response_cache)}")

    def _update_conversation(self, conversation_id: str,
                             current_turn_messages: List[ChatMessage],
                             assistant_response_content: str) -> None:
        """Updated die Konversationshistorie."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
            logger.info(f"Started new conversation with ID: {conversation_id}")

        # Add messages from the current turn if they are not already in history
        # This assumes `current_turn_messages` contains the system/user messages leading to this response
        for msg in current_turn_messages:
            # A simple check to avoid duplicate message objects if chat() is called repeatedly with same history
            # More robust would be checking message IDs if they exist.
            if not self.conversations[conversation_id] or msg.content != self.conversations[conversation_id][-1].content:
                 self.conversations[conversation_id].append(msg)


        # Add the assistant's response
        assistant_message = ChatMessage(
            role="assistant",
            content=assistant_response_content,
            timestamp=datetime.now().isoformat()
        )
        self.conversations[conversation_id].append(assistant_message)

        # Limit conversation history length (e.g., last 20 messages or N turns)
        max_history_messages = 20 # Example: keep last 20 messages (10 turns)
        if len(self.conversations[conversation_id]) > max_history_messages:
            self.conversations[conversation_id] = self.conversations[conversation_id][-max_history_messages:]
            logger.debug(f"Conversation '{conversation_id}' history truncated to last {max_history_messages} messages.")
        logger.info(f"Conversation '{conversation_id}' updated. Total messages: {len(self.conversations[conversation_id])}")


    def _extract_context_summary(self, context_documents: List[Dict[str, Any]]) -> Optional[str]:
        """Extrahiert eine kurze Zusammenfassung aus den Kontextdokumenten für Logging/Metadaten."""
        if not context_documents:
            return None
        
        summaries = []
        for i, doc in enumerate(context_documents[:2]):  # Summary from top 2 docs
            content = doc.get('content', doc.get('text', ''))
            title = doc.get('metadata', {}).get('title', f'Doc{i+1}')
            summary_part = content[:75] + "..." if len(content) > 75 else content
            summaries.append(f"{title}: {summary_part}")
        
        return " | ".join(summaries) if summaries else None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Liefert detaillierte Performance-Statistiken des LLM Handlers."""
        avg_processing_time_per_req = (self.total_processing_time / self.total_requests) if self.total_requests > 0 else 0
        avg_tokens_per_req = (self.total_tokens_generated / self.total_requests) if self.total_requests > 0 else 0
        cache_total_lookups = self.cache_hits + self.cache_misses
        cache_hit_rate_val = (self.cache_hits / cache_total_lookups) if cache_total_lookups > 0 else 0.0
        
        # Tokens per second based on actual generation time (total_processing_time excludes cache lookup time)
        tokens_per_second_generation = (self.total_tokens_generated / self.total_processing_time) if self.total_processing_time > 0 else 0.0
        
        stats = {
            "current_model_name": self.model_name or "Not loaded",
            "total_requests_processed": self.total_requests,
            "total_tokens_generated_estimate": self.total_tokens_generated,
            "total_llm_processing_time_seconds": round(self.total_processing_time, 3),
            "average_llm_processing_time_per_request_seconds": round(avg_processing_time_per_req, 3),
            "average_tokens_generated_per_request_estimate": round(avg_tokens_per_req, 1),
            "estimated_tokens_per_second_generation": round(tokens_per_second_generation, 1),
            "batch_generation_calls": self.batch_count,
            "response_cache_status": "enabled" if self.config.cache_responses and self.response_cache is not None else "disabled",
            "response_cache_hits": self.cache_hits,
            "response_cache_misses": self.cache_misses,
            "response_cache_hit_rate": round(cache_hit_rate_val, 3),
            "response_cache_current_size": len(self.response_cache) if self.response_cache is not None else 0,
            "response_cache_max_size": self.config.cache_size if self.config.cache_responses else 0,
            "active_conversations_tracked": len(self.conversations),
            "supported_model_profiles_count": len(self.supported_models),
            "mlx_parallm_available": MLX_PARALLM_AVAILABLE
        }
        logger.info(f"Performance stats retrieved for model '{self.model_name}'.")
        return stats

    def clear_cache(self) -> None:
        """Leert den Response Cache und setzt Cache-Statistiken zurück."""
        if self.response_cache is not None:
            self.response_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Response cache cleared and stats reset.")

    def clear_conversations(self) -> None:
        """Leert die gesamte Konversationshistorie."""
        self.conversations.clear()
        logger.info("All conversation histories cleared.")

    async def benchmark(self, num_prompts: int = 20, single_comparison_count: int = 5) -> Dict[str, float]:
        """
        Führt einen Performance-Benchmark durch für Single- vs. Batch-Generierung.
        Verwendet das aktuell konfigurierte/initialisierte Modell.
        """
        if not self.model or not self.tokenizer:
            logger.error("Benchmark cannot run: Model not initialized.")
            await self.initialize() # Attempt to initialize default model
            if not self.model or not self.tokenizer:
                 return {"error": "Model not initialized for benchmark."} # type: ignore

        logger.info(f"Running LLM Handler Benchmark with model '{self.model_name}'...")
        logger.info(f"Total prompts for batch: {num_prompts}, Prompts for single comparison: {single_comparison_count}")

        test_prompts_list = [
            "What is the main idea behind the theory of relativity?",
            "Explain the process of photosynthesis in detail.",
            "Write a short story about a robot discovering emotions.",
            "What are the ethical implications of artificial intelligence?",
            "Describe three common data structures and their use cases.",
        ] * (num_prompts // 5 or 1) # Ensure we have enough prompts
        test_prompts_list = test_prompts_list[:num_prompts]


        llm_requests = [
            LLMRequest(
                prompt=p, user_id="benchmark_user",
                max_tokens=self.config.max_tokens // 2, # Shorter responses for benchmark speed
                temperature=self.config.temperature
            ) for p in test_prompts_list
        ]

        # Clear cache for a fair benchmark run
        original_cache_setting = self.config.cache_responses
        self.config.cache_responses = False # Disable cache for benchmark
        if self.response_cache is not None: self.clear_cache()
        logger.info("Cache temporarily disabled for benchmark.")

        # --- Benchmark single generation (for comparison) ---
        single_gen_times = []
        single_tokens_generated = 0
        logger.info(f"Benchmarking single generation with {single_comparison_count} prompts...")
        single_start_time = time.monotonic()
        for i in range(min(single_comparison_count, len(llm_requests))):
            req = llm_requests[i]
            start_t = time.monotonic()
            response_obj = await self.generate_single(req)
            single_gen_times.append(time.monotonic() - start_t)
            single_tokens_generated += response_obj.token_count
        single_total_time = time.monotonic() - single_start_time
        avg_single_time = sum(single_gen_times) / len(single_gen_times) if single_gen_times else 0
        single_req_per_sec = len(single_gen_times) / single_total_time if single_total_time > 0 else 0
        single_tok_per_sec = single_tokens_generated / single_total_time if single_total_time > 0 else 0
        logger.info(f"Single generation: {single_req_per_sec:.2f} req/s, {single_tok_per_sec:.1f} tok/s (avg per req: {avg_single_time:.3f}s)")


        # --- Benchmark batch generation ---
        logger.info(f"Benchmarking batch generation with {len(llm_requests)} prompts...")
        batch_start_time = time.monotonic()
        batch_response_objects = await self.generate_batch(llm_requests)
        batch_total_time = time.monotonic() - batch_start_time
        
        batch_tokens_generated = sum(r.token_count for r in batch_response_objects)
        batch_req_per_sec = len(batch_response_objects) / batch_total_time if batch_total_time > 0 else 0
        batch_tok_per_sec = batch_tokens_generated / batch_total_time if batch_total_time > 0 else 0
        avg_batch_time_per_req_equivalent = batch_total_time / len(batch_response_objects) if batch_response_objects else 0
        logger.info(f"Batch generation: {batch_req_per_sec:.2f} req/s, {batch_tok_per_sec:.1f} tok/s (avg per req in batch: {avg_batch_time_per_req_equivalent:.3f}s)")

        # Restore original cache setting and clear benchmark data from cache (if any made it despite disabling)
        self.config.cache_responses = original_cache_setting
        if self.response_cache is not None: self.clear_cache() # Clear again to remove any benchmark items
        logger.info(f"Cache setting restored to: {'Enabled' if self.config.cache_responses else 'Disabled'}.")

        speedup_factor = (avg_single_time / avg_batch_time_per_req_equivalent) if avg_batch_time_per_req_equivalent > 0 and avg_single_time > 0 else 0.0

        results = {
            "model_benchmarked": str(self.model_name),
            "num_prompts_single_comparison": len(single_gen_times),
            "total_time_single_comparison_seconds": round(single_total_time, 3),
            "avg_time_per_single_request_seconds": round(avg_single_time, 3),
            "single_requests_per_second": round(single_req_per_sec, 2),
            "single_tokens_per_second_estimate": round(single_tok_per_sec, 1),
            "num_prompts_batch": len(batch_response_objects),
            "total_time_batch_seconds": round(batch_total_time, 3),
            "avg_time_per_request_in_batch_seconds": round(avg_batch_time_per_req_equivalent, 3),
            "batch_requests_per_second": round(batch_req_per_sec, 2),
            "batch_tokens_per_second_estimate": round(batch_tok_per_sec, 1),
            "estimated_batch_speedup_factor": round(speedup_factor, 2),
            "total_tokens_generated_single_comp": single_tokens_generated,
            "total_tokens_generated_batch": batch_tokens_generated,
        }
        logger.info(f"Benchmark results: {results}")
        return results


# Example Usage (Updated)
async def example_usage():
    """Beispiele für die Verwendung des MLXLLMHandler."""
    logger.info("--- Starting MLXLLMHandler Example Usage ---")

    # Initialize with a specific configuration
    llm_config = LLMConfig(
        model_path="mlx-community/gemma-2-9b-it-4bit", # Ensure this model path is valid for mlx_parallm
        batch_size=8, # Optimal batch size for this model (example)
        max_batch_size=32,
        cache_responses=True,
        cache_size=100,
        format_prompts=True # Important for Gemma models if mlx_parallm supports it
    )
    llm_handler = MLXLLMHandler(llm_config)

    try:
        # It's good practice to initialize explicitly if you need to ensure model is ready
        await llm_handler.initialize()
        if not llm_handler.model:
            logger.error("LLM Handler failed to initialize model in example. Exiting.")
            return

        # --- Single Generation Example ---
        logger.info("\n--- Single Generation Example ---")
        single_req = LLMRequest(
            prompt="What are three key benefits of using the MLX framework on Apple Silicon?",
            user_id="user_single_example",
            max_tokens=150,
            temperature=0.6
        )
        single_resp = await llm_handler.generate_single(single_req)
        logger.info(f"Single Response for '{single_req.prompt[:30]}...':\n{single_resp.response}")
        logger.info(f"Processing time: {single_resp.processing_time:.3f}s, Cached: {single_resp.cached}, Tokens: {single_resp.token_count}")

        # --- Batch Generation Example ---
        logger.info("\n--- Batch Generation Example ---")
        batch_req_list = [
            LLMRequest(prompt="Explain the concept of 'unified memory'.", user_id="user_batch1", max_tokens=100),
            LLMRequest(prompt="List two advantages of 4-bit quantization for LLMs.", user_id="user_batch2", max_tokens=120),
            LLMRequest(prompt="What is a transformer model in the context of AI?", user_id="user_batch3", max_tokens=180)
        ]
        batch_resp_list = await llm_handler.generate_batch(batch_req_list)
        logger.info(f"Batch generated {len(batch_resp_list)} responses.")
        for i, resp in enumerate(batch_resp_list):
            logger.info(f"Batch Response {i+1} (User: {resp.user_id}): {resp.response[:100]}... (Cached: {resp.cached})")

        # --- Chat Interface Example ---
        logger.info("\n--- Chat Interface Example ---")
        chat_conversation_id = "conv_example_123"
        initial_messages = [
            ChatMessage(role="system", content="You are an expert on Apple technologies and machine learning."),
            ChatMessage(role="user", content="Hello! Can you tell me about MLX?")
        ]
        chat_resp1 = await llm_handler.chat(initial_messages, "user_chat_example", chat_conversation_id)
        logger.info(f"Chat Response 1 (ConvID: {chat_conversation_id}):\n{chat_resp1.response}")

        follow_up_messages = llm_handler.conversations.get(chat_conversation_id, []).copy() # Get current history
        follow_up_messages.append(ChatMessage(role="user", content="How does it compare to PyTorch for on-device inference?"))
        
        chat_resp2 = await llm_handler.chat(follow_up_messages, "user_chat_example", chat_conversation_id)
        logger.info(f"Chat Response 2 (ConvID: {chat_conversation_id}):\n{chat_resp2.response}")


        # --- RAG-optimized Generation Example ---
        logger.info("\n--- RAG Generation Example ---")
        rag_query = "What specific MLX features enhance performance on M-series chips?"
        rag_context_docs = [
            {"content": "MLX leverages Apple Silicon's unified memory architecture to avoid data copying between CPU and GPU, significantly speeding up operations.", "metadata": {"title": "MLX Unified Memory Advantage"}},
            {"content": "The framework includes optimized kernels for Metal, Apple's graphics API, ensuring efficient GPU utilization for ML tasks.", "metadata": {"title": "MLX Metal Kernels"}},
            {"content": "MLX supports efficient an array framework that is similar to NumPy and composable function transformations for automatic differentiation, vectorization, and parallelization.", "metadata": {"title": "MLX Array Framework"}}
        ]
        rag_system_prompt = "You are an AI assistant specializing in MLX. Provide a detailed answer based *only* on the provided context documents, citing them as 'Source: [Title]'."
        
        rag_llm_response = await llm_handler.rag_generate(
            query=rag_query,
            context_documents=rag_context_docs,
            user_id="user_rag_example",
            system_prompt=rag_system_prompt,
            max_tokens=250
        )
        logger.info(f"RAG Response for '{rag_query}':\n{rag_llm_response.response}")

        # --- Performance Statistics ---
        logger.info("\n--- Performance Statistics ---")
        perf_stats = llm_handler.get_performance_stats()
        logger.info(f"Current Performance Stats: {json.dumps(perf_stats, indent=2)}")

        # --- Benchmark ---
        logger.info("\n--- Running Benchmark ---")
        # Reduce counts for quicker example run
        benchmark_data = await llm_handler.benchmark(num_prompts=10, single_comparison_count=3)
        logger.info(f"Benchmark Results: {json.dumps(benchmark_data, indent=2)}")
        
        # --- Clear Cache and Conversations ---
        llm_handler.clear_cache()
        logger.info("Response cache cleared for example.")
        llm_handler.clear_conversations()
        logger.info("Conversation history cleared for example.")

    except ImportError as ie:
        logger.error(f"Example usage failed due to missing import: {ie}. Ensure mlx and mlx_parallm are installed.")
    except Exception as e:
        logger.error(f"An error occurred during example usage: {e}", exc_info=True)

if __name__ == "__main__":
    # This example assumes that the necessary models (e.g., "mlx-community/gemma-2-9b-it-4bit")
    # are accessible by the `mlx_parallm.utils.load` function.
    # If MLX_PARALLM_AVAILABLE is False, the example will likely fail at model loading
    # unless the mock `load` function is sufficient for basic script execution without error.
    logger.info("Starting MLXLLMHandler example execution.")
    # asyncio.run(example_usage()) # Uncomment to run
    logger.info("Example usage commented out. Uncomment to run with mlx and mlx_parallm properly set up.")

