"""
MLX Research Assistant
Intelligente Web Research mit RAG Integration
Kombiniert Web Scraping, Content Processing und MLX Components
"""

import asyncio
import aiohttp
import time
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta # --- timedelta WAS NOT USED, kept for now ---
import json
import aiofiles
from urllib.parse import urljoin, urlparse, quote # --- quote WAS NOT USED, kept for now ---
import ssl

# Web scraping libraries
from bs4 import BeautifulSoup
import feedparser # --- WAS IMPORTED ---
import newspaper # --- WAS IMPORTED ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# import requests # --- WAS IMPORTED but aiohttp is used for async, removing 'requests' unless specifically needed for a sync part ---

# MLX Components Integration
# --- Assuming these are correctly set up and available in the environment ---
from mlx_components.embedding_engine import MLXEmbeddingEngine, EmbeddingConfig
from mlx_components.vector_store import MLXVectorStore, VectorStoreConfig # --- VectorStoreConfig was not used by name, kept for consistency ---
from mlx_components.llm_handler import MLXLLMHandler, LLMConfig, LLMRequest
from mlx_components.rerank_engine import MLXRerankEngine, ReRankConfig
from tools.document_processor import MLXDocumentProcessor, ProcessingConfig


@dataclass
class ResearchConfig:
    """Konfiguration für Research Assistant"""
    max_search_results: int = 20
    max_pages_per_domain: int = 5
    request_timeout: int = 30 # seconds
    rate_limit_delay: float = 1.0 # seconds
    user_agent: str = "MLX-Research-Assistant/1.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)" # --- Enhanced User-Agent ---
    enable_javascript: bool = False # Selenium usage, can be slow
    max_content_length: int = 500000 # --- Increased from 50k to 500k for more comprehensive extraction before truncation ---
    extract_links: bool = True
    follow_redirects: bool = True # aiohttp handles this by default up to a limit
    verify_ssl: bool = True # For aiohttp session
    embedding_model: str = "mlx-community/gte-small"
    llm_model: str = "mlx-community/gemma-2-9b-it-4bit" # Example model
    auto_summarize: bool = True
    save_raw_content: bool = False # For WebContent, consider if needed due to memory
    exclude_domains: List[str] = None
    include_social_media: bool = False
    selenium_wait_timeout: int = 10 # seconds to wait for elements with Selenium

    def __post_init__(self):
        if self.exclude_domains is None:
            self.exclude_domains = [
                "facebook.com", "twitter.com", "instagram.com", "tiktok.com",
                "linkedin.com", "pinterest.com", "reddit.com", "youtube.com" # Added youtube
            ] if not self.include_social_media else []

@dataclass
class SearchQuery:
    """Suchanfrage Definition"""
    query: str
    user_id: str
    search_engines: List[str] = None
    max_results: int = 10 # Max results per engine
    time_filter: str = "all"  # all, day, week, month, year (Note: not all scrapers support this)
    language: str = "en"
    country: str = "us" # For search localization (Note: not all scrapers support this)
    safe_search: bool = True # (Note: not all scrapers support this)
    context: Optional[str] = None
    follow_up_questions: Optional[List[str]] = None # --- MODIFIED: made Optional ---

    def __post_init__(self):
        if self.search_engines is None:
            # Defaulting to potentially more reliable / less block-prone options first
            self.search_engines = ["duckduckgo", "bing", "google"] # User can override

@dataclass
class WebContent:
    """Extrahierter Web-Content"""
    url: str
    title: str
    content: str # Main textual content
    summary: Optional[str] # AI Generated or extracted
    author: Optional[str] # Or list of authors
    publish_date: Optional[datetime]
    domain: str
    language: Optional[str] # --- MODIFIED: Made Optional, newspaper might not always detect ---
    word_count: int
    links: List[str] # Extracted internal/external links
    images: List[str] # URLs of key images
    metadata: Dict[str, Any] # Other metadata like keywords, site name, etc.
    extraction_time: datetime
    content_hash: str # MD5 hash of the 'content' field
    embedding: Optional[List[float]] = None
    raw_html: Optional[str] = None # --- ADDED: Optional raw HTML if config.save_raw_content ---
    source_engine: Optional[str] = None # --- ADDED: Which search engine found this URL ---

@dataclass
class ResearchResult:
    """Forschungsergebnis"""
    query: str
    search_results: List[WebContent] # Top N relevant WebContent objects
    synthesized_answer: str
    sources_used: List[Dict[str,str]] # --- MODIFIED: Simplified to dict for direct use --- (title, url, domain)
    confidence_score: float # 0.0 to 1.0
    research_time: float # seconds
    follow_up_questions: List[str]
    related_topics: List[str]
    user_id: str
    timestamp: datetime
    context_used: Optional[str] = None

class MLXResearchAssistant:
    """
    High-Performance Research Assistant für MLX Ecosystem
    (Refer to original docstring for features)
    """

    def __init__(self,
                 config: ResearchConfig = None,
                 embedding_engine: Optional[MLXEmbeddingEngine] = None, # --- MODIFIED: Made Optional explicit ---
                 vector_store: Optional[MLXVectorStore] = None,     # --- MODIFIED: Made Optional explicit ---
                 llm_handler: Optional[MLXLLMHandler] = None,       # --- MODIFIED: Made Optional explicit ---
                 rerank_engine: Optional[MLXRerankEngine] = None,     # --- MODIFIED: Made Optional explicit ---
                 document_processor: Optional[MLXDocumentProcessor] = None): # --- MODIFIED: Made Optional explicit ---
        self.config = config or ResearchConfig()
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.llm_handler = llm_handler
        self.rerank_engine = rerank_engine
        self.document_processor = document_processor

        self.session: Optional[aiohttp.ClientSession] = None
        self.chrome_driver_path: Optional[str] = None # --- ADDED: Path for chromedriver if not in PATH ---
        # --- ADDED: Selenium WebDriver instance, manage its lifecycle if enable_javascript is True ---
        self._selenium_driver: Optional[webdriver.Chrome] = None


        self.total_queries = 0
        self.total_pages_extracted = 0
        self.total_research_time = 0.0
        self.successful_extractions = 0
        self.failed_extractions = 0

        # --- MODIFIED: Cache could be more sophisticated (e.g., LRU, TTL) ---
        self.content_cache: Dict[str, WebContent] = {} # Cache key: content_hash or url_hash
        # self.embedding_cache = {} # Embedding cache is handled by MLXEmbeddingEngine typically

        self.search_engines = {
            "duckduckgo": self._search_duckduckgo,
            "bing": self._search_bing_html, # Defaulting to HTML version for Bing
            "google": self._search_startpage, # Defaulting to Startpage for Google results
            # To use actual APIs for Bing/Google, they'd need API key handling
        }

        # --- Order of extractors can matter. Newspaper first, then BS4, then Selenium as last resort ---
        self.extractors = {
            "feed": self._extract_with_feedparser, # --- ADDED: Feedparser first ---
            "newspaper": self._extract_with_newspaper,
            "beautifulsoup": self._extract_with_bs4,
            "selenium": self._extract_with_selenium,
        }

    async def initialize(self) -> None:
        """Initialisiert Research Assistant mit allen Dependencies"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            # --- MODIFIED: SSL context handling ---
            ssl_context = ssl.create_default_context() if self.config.verify_ssl else False
            connector = aiohttp.TCPConnector(
                limit=30, # --- Reduced global limit ---
                limit_per_host=5, # --- Reduced per-host limit ---
                ssl=ssl_context,
                enable_cookies=False # --- ADDED: Disable cookies for general scraping unless needed ---
            )
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={"User-Agent": self.config.user_agent}
            )

        # --- MODIFIED: Initialize MLX components only if they are None (allowing pre-configured instances) ---
        if self.embedding_engine is None and self.config.embedding_model: # Check if model is configured
            embedding_config = EmbeddingConfig(
                model_path=self.config.embedding_model, batch_size=16, cache_embeddings=True
            )
            self.embedding_engine = MLXEmbeddingEngine(embedding_config)
            await self.embedding_engine.initialize()

        if self.llm_handler is None and self.config.llm_model: # Check if model is configured
            llm_config = LLMConfig(
                model_path=self.config.llm_model, batch_size=1, cache_responses=True # Batch size 1 for research often
            )
            self.llm_handler = MLXLLMHandler(llm_config)
            await self.llm_handler.initialize()

        if self.rerank_engine is None: # Rerank engine might have default internal models or config
            rerank_config = ReRankConfig(top_k=10, diversity_factor=0.3) # Default config
            self.rerank_engine = MLXRerankEngine(rerank_config)
            # Assuming rerank_engine.initialize() is not always needed or handled internally
            # await self.rerank_engine.initialize() # Uncomment if your reranker needs async init

        if self.document_processor is None:
            doc_config = ProcessingConfig(
                chunk_size=800, # --- Reduced chunk size slightly ---
                auto_summarize=self.config.auto_summarize, # Use research config
                embedding_model=self.config.embedding_model
            )
            self.document_processor = MLXDocumentProcessor(doc_config, self.embedding_engine)
            # Assuming document_processor.initialize() is not needed

        # --- ADDED: Initialize Selenium WebDriver if enabled, called once ---
        if self.config.enable_javascript and self._selenium_driver is None:
            await self._initialize_selenium_driver()


    async def _initialize_selenium_driver(self): # --- ADDED ---
        """Initializes the Selenium WebDriver."""
        if self._selenium_driver is not None:
            return
        print("Initializing Selenium WebDriver (headless)...")
        try:
            loop = asyncio.get_event_loop()
            def setup_driver():
                chrome_options = Options()
                chrome_options.add_argument("--headless")
                chrome_options.add_argument("--no-sandbox")
                chrome_options.add_argument("--disable-dev-shm-usage")
                chrome_options.add_argument("--disable-gpu")
                chrome_options.add_argument(f"--user-agent={self.config.user_agent}")
                chrome_options.add_argument("--blink-settings=imagesEnabled=false") # Disable images
                # Add more options to reduce resource usage if needed
                if self.chrome_driver_path:
                    service = webdriver.chrome.service.Service(executable_path=self.chrome_driver_path)
                    return webdriver.Chrome(service=service, options=chrome_options)
                else:
                    # Assumes chromedriver is in PATH
                    return webdriver.Chrome(options=chrome_options)
            self._selenium_driver = await loop.run_in_executor(None, setup_driver)
            print("Selenium WebDriver initialized.")
        except Exception as e:
            print(f"⚠️ Failed to initialize Selenium WebDriver: {e}. JavaScript rendering will be disabled.")
            self.config.enable_javascript = False # Disable if init fails
            self._selenium_driver = None


    async def close(self) -> None:
        """Schließt HTTP Session und Resourcen"""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
        # --- ADDED: Quit Selenium driver ---
        if self._selenium_driver:
            print("Closing Selenium WebDriver...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._selenium_driver.quit)
            self._selenium_driver = None
            print("Selenium WebDriver closed.")

    async def research(self, search_query: SearchQuery) -> ResearchResult:
        """Hauptfunktion: Führt komplette Research durch"""
        start_time = time.time()
        await self.initialize()
        print(f"🔍 Starting research for query: '{search_query.query}' by user '{search_query.user_id}'")

        try:
            search_links_with_source_engine = await self._multi_engine_search(search_query) # Returns list of (url, title, snippet, engine)
            print(f"📄 Found {len(search_links_with_source_engine)} initial search links.")

            # --- Filter search_links by exclude_domains before extraction ---
            filtered_links = []
            for res_dict in search_links_with_source_engine:
                url = res_dict.get("url")
                if url:
                    domain = urlparse(url).netloc.lower()
                    if not any(excluded_domain in domain for excluded_domain in self.config.exclude_domains):
                        filtered_links.append(res_dict)
                    else:
                        print(f"🚫 Excluding URL from banned domain: {url}")
            
            print(f"🔗 Filtered to {len(filtered_links)} links after domain exclusion.")


            web_contents = await self._extract_web_contents(filtered_links, search_query) # Pass search_query for context
            print(f"📃 Extracted {len(web_contents)} web pages successfully.")

            if not web_contents: # --- ADDED: Handle no content ---
                print("No content extracted. Returning early.")
                # Fallback for no content
                return ResearchResult(
                    query=search_query.query, search_results=[],
                    synthesized_answer="Could not find and extract relevant information for your query.",
                    sources_used=[], confidence_score=0.1, research_time=time.time() - start_time,
                    follow_up_questions=[], related_topics=[], user_id=search_query.user_id,
                    timestamp=datetime.utcnow(), context_used=search_query.context
                )

            # Processing and embedding are now part of _extract_single_content if enabled,
            # or could be a separate step if document_processor is used more centrally.
            # For this flow, we assume WebContent objects might already have summary/embedding if done during extraction.
            # If not, _process_contents can be called here.
            # processed_contents = await self._process_contents(web_contents, search_query.user_id) # Assuming this is more for chunking etc.

            # Re-rank based on query and (summary or content)
            reranked_contents = await self._rerank_contents(search_query.query, web_contents)
            print(f"📊 Re-ranked to top {len(reranked_contents)} relevant pages.")

            synthesis_result = await self._synthesize_answer(search_query, reranked_contents)
            print(f"💡 Answer synthesized. Length: {len(synthesis_result['answer'])} chars, Confidence: {synthesis_result['confidence']:.2f}")

            follow_ups = await self._generate_follow_up_questions(search_query.query, synthesis_result["answer"])
            related_topics = await self._extract_related_topics(reranked_contents) # Use reranked for more relevance

            research_time_taken = time.time() - start_time
            self.total_queries += 1
            self.total_pages_extracted += self.successful_extractions # successful_extractions is updated in _extract_single_content
            self.total_research_time += research_time_taken

            final_result = ResearchResult(
                query=search_query.query,
                search_results=reranked_contents, # Store the re-ranked, most relevant WebContent
                synthesized_answer=synthesis_result["answer"],
                sources_used=synthesis_result["sources"], # List of dicts {title, url, domain}
                confidence_score=synthesis_result["confidence"],
                research_time=research_time_taken,
                follow_up_questions=follow_ups,
                related_topics=related_topics,
                user_id=search_query.user_id,
                timestamp=datetime.utcnow(), # --- Use UTC for consistency ---
                context_used=search_query.context
            )
            print(f"✅ Research completed in {research_time_taken:.2f}s for query: '{search_query.query}'")
            return final_result

        except Exception as e:
            print(f"❌ Major research error for query '{search_query.query}': {e}")
            import traceback
            traceback.print_exc()
            # Return a meaningful error result
            return ResearchResult(
                query=search_query.query, search_results=[],
                synthesized_answer=f"An error occurred during research: {str(e)}",
                sources_used=[], confidence_score=0.0, research_time=time.time() - start_time,
                follow_up_questions=[], related_topics=[], user_id=search_query.user_id,
                timestamp=datetime.utcnow(), context_used=search_query.context
            )


    async def quick_search(self,
                           query: str,
                           user_id: str,
                           max_results: int = 5) -> List[WebContent]:
        """Schnelle Suche ohne vollständige Synthese, nur Extraktion."""
        print(f"🚀 Performing quick search for: '{query}'")
        # --- MODIFIED: Use a more focused search engine for quick search if desired ---
        search_query_obj = SearchQuery(
            query=query, user_id=user_id, max_results=max_results,
            search_engines=["duckduckgo"] # Example: only DDG for speed
        )
        await self.initialize()
        
        # --- MODIFIED: Call _multi_engine_search which returns list of dicts ---
        search_links = await self._multi_engine_search(search_query_obj)
        if not search_links:
            print("Quick search yielded no links.")
            return []
        
        print(f"Quick search found {len(search_links)} links. Extracting content...")
        # --- MODIFIED: Pass the SearchQuery object to _extract_web_contents ---
        web_contents_list = await self._extract_web_contents(search_links, search_query_obj)
        print(f"Quick search extracted {len(web_contents_list)} pages.")
        return web_contents_list


    async def _multi_engine_search(self, search_query: SearchQuery) -> List[Dict[str, str]]:
        """Sucht über mehrere Suchmaschinen parallel. Returns list of dicts: {title, url, snippet, engine}."""
        all_raw_results: List[Dict[str, str]] = [] # --- Store dicts directly ---
        tasks = []

        # --- MODIFIED: Pass more parameters from SearchQuery to individual search methods if they support them ---
        for engine_name in search_query.search_engines:
            if engine_name in self.search_engines:
                # Individual search methods currently only take query and max_results.
                # They could be enhanced to take lang, country, time_filter etc.
                task = self.search_engines[engine_name](search_query.query, search_query.max_results)
                tasks.append((engine_name, task))
            else:
                print(f"⚠️ Unknown search engine specified: {engine_name}")

        if not tasks: return []

        gathered_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        for i, (engine_name, _) in enumerate(tasks):
            if isinstance(gathered_results[i], Exception):
                print(f"⚠️ Search engine '{engine_name}' failed: {gathered_results[i]}")
            elif gathered_results[i]:
                engine_specific_results = gathered_results[i]
                print(f"🔍 Engine '{engine_name}' returned {len(engine_specific_results)} results.")
                # Add engine info to each result dict
                for res_dict in engine_specific_results:
                    res_dict["engine"] = engine_name 
                all_raw_results.extend(engine_specific_results)
            else:
                 print(f"🔍 Engine '{engine_name}' returned no results.")


        # Deduplicate results by URL, preferring results from earlier engines in config or those with longer snippets
        seen_urls: Set[str] = set()
        unique_deduped_results: List[Dict[str,str]] = []
        for res_dict in all_raw_results:
            url = res_dict.get("url")
            if url and url not in seen_urls:
                # Basic normalization of URL to improve deduplication
                parsed_url = urlparse(url)
                # Remove 'www.' prefix, ensure scheme, remove fragment
                normalized_url = f"{parsed_url.scheme or 'http'}://{parsed_url.netloc.replace('www.','')}{parsed_url.path}{('?' + parsed_url.query if parsed_url.query else '')}"
                
                if normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    unique_deduped_results.append(res_dict)
        
        # Further refine or sort if needed. For now, just limit.
        return unique_deduped_results[:self.config.max_search_results]


    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """DuckDuckGo Suche (API-frei). Returns list of dicts {title, url, snippet}."""
        # --- This method was mostly fine, minor logging added ---
        results: List[Dict[str, str]] = []
        try:
            # DuckDuckGo Lite HTML for more results
            results.extend(await self._search_duckduckgo_html(query, max_results))
            
            # Try Instant Answer API as a supplement if HTML yields few results
            if len(results) < max_results // 2 : # If HTML results are too few
                print("DDG HTML results were few, trying Instant Answer API...")
                search_url = "https://api.duckduckgo.com/"
                params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
                if not self.session or self.session.closed: await self.initialize() # Ensure session
                
                async with self.session.get(search_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response: # Shorter timeout for API
                    if response.status == 200:
                        data = await response.json()
                        api_res_count = 0
                        # RelatedTopics often contain the most useful web links
                        for topic in data.get("RelatedTopics", []):
                            if isinstance(topic, dict) and "FirstURL" in topic and topic["FirstURL"]:
                                results.append({
                                    "title": topic.get("Text", "").split(" - ", 1)[0], # Basic title parsing
                                    "url": topic["FirstURL"],
                                    "snippet": topic.get("Text", ""), # Full text as snippet
                                })
                                api_res_count +=1
                                if len(results) >= max_results: break
                            # Also check for Topics within RelatedTopics (nested structure)
                            elif isinstance(topic, dict) and "Topics" in topic and isinstance(topic["Topics"], list):
                                for sub_topic in topic["Topics"]:
                                     if isinstance(sub_topic, dict) and "FirstURL" in sub_topic and sub_topic["FirstURL"]:
                                        results.append({
                                            "title": sub_topic.get("Text", "").split(" - ", 1)[0],
                                            "url": sub_topic["FirstURL"],
                                            "snippet": sub_topic.get("Text", ""),
                                        })
                                        api_res_count +=1
                                        if len(results) >= max_results: break
                                if len(results) >= max_results: break
                        # Also check "Results" for direct links (less common in Instant Answer)
                        for res_item in data.get("Results", []):
                             if isinstance(res_item, dict) and "FirstURL" in res_item and res_item["FirstURL"]:
                                results.append({
                                    "title": res_item.get("Text", "").split(" - ", 1)[0],
                                    "url": res_item["FirstURL"],
                                    "snippet": res_item.get("Text", ""),
                                })
                                api_res_count +=1
                                if len(results) >= max_results: break
                        print(f"DDG Instant Answer API added {api_res_count} results.")
                    else:
                        print(f"DuckDuckGo API request failed with status: {response.status}")
        except asyncio.TimeoutError:
            print(f"DuckDuckGo API search timed out for query: {query}")
        except Exception as e:
            print(f"DuckDuckGo search encountered an error: {e}")
        
        # Deduplicate again after combining sources, just in case
        seen_urls = set()
        final_results = []
        for r in results:
            if r["url"] not in seen_urls:
                final_results.append(r)
                seen_urls.add(r["url"])
        return final_results[:max_results]


    async def _search_duckduckgo_html(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """DuckDuckGo HTML Suche. Returns list of dicts {title, url, snippet}."""
        # --- This method was mostly fine, minor logging ---
        results: List[Dict[str, str]] = []
        search_url = "https://html.duckduckgo.com/html/" # Using the HTML version
        params = {"q": query}
        # DDG HTML version is sensitive to user agents, use a common one.
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"}
        
        try:
            if not self.session or self.session.closed: await self.initialize() # Ensure session
            async with self.session.get(search_url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Select result links; class names might change, inspect DDG HTML output
                    # Common pattern: results are in divs with class "result" or "web-result"
                    result_divs = soup.find_all('div', class_=['result', 'web-result', 'results_links_deep']) # Try multiple class names
                    
                    for res_div in result_divs:
                        title_tag = res_div.find('a', class_=['result__a', 'result-link'])
                        snippet_tag = res_div.find('a', class_=['result__snippet', 'result-snippet']) # Snippet is also a link sometimes
                        if not snippet_tag: # Fallback for snippet
                             snippet_tag = res_div.find('div', class_='result__snippet')


                        if title_tag and title_tag.get('href'):
                            url = title_tag['href']
                            # DDG HTML links are often redirects, try to clean them if needed
                            # e.g., /l/?kh=-1&uddg=...
                            if url.startswith("/l/"):
                                parsed_ddg_url = urlparse(url)
                                query_params = dict(qc.split("=") for qc in parsed_ddg_url.query.split("&") if "=" in qc)
                                actual_url = query_params.get("uddg")
                                if actual_url:
                                    url = actual_url
                                else: # If cannot resolve, skip this DDG redirect unless it's the only option
                                    # print(f"Could not resolve DDG redirect: {title_tag['href']}")
                                    continue # Skip these complex redirects for now

                            results.append({
                                "title": title_tag.get_text(strip=True),
                                "url": url,
                                "snippet": snippet_tag.get_text(strip=True) if snippet_tag else ""
                            })
                            if len(results) >= max_results:
                                break
                    print(f"DuckDuckGo HTML search found {len(results)} links for '{query}'.")
                else:
                    print(f"DuckDuckGo HTML search failed for '{query}' with status: {response.status}")
        except asyncio.TimeoutError:
            print(f"DuckDuckGo HTML search timed out for query: {query}")
        except Exception as e:
            print(f"DuckDuckGo HTML search error for '{query}': {e}")
        return results[:max_results] # Ensure max_results respected


    async def _search_bing_html(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Bing HTML Suche. Returns list of dicts {title, url, snippet}."""
        # --- This method was mostly fine, added timeout, ensure session ---
        results: List[Dict[str, str]] = []
        search_url = "https://www.bing.com/search"
        params = {"q": query, "count": str(max_results), "format": "html"} # 'count' for Bing
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36"}
        try:
            if not self.session or self.session.closed: await self.initialize()
            async with self.session.get(search_url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    # Bing's result list items often have class 'b_algo'
                    for res_item in soup.find_all('li', class_='b_algo', limit=max_results):
                        title_tag = res_item.find('h2')
                        link_tag = title_tag.find('a') if title_tag else None
                        snippet_tag = res_item.find('div', class_='b_caption') # Snippet is often in a div with class 'b_caption'
                        if not snippet_tag: # Fallback
                             snippet_tag = res_item.find('p')


                        if link_tag and link_tag.get('href'):
                            results.append({
                                "title": link_tag.get_text(strip=True),
                                "url": link_tag['href'],
                                "snippet": snippet_tag.get_text(strip=True) if snippet_tag else ""
                            })
                    print(f"Bing HTML search found {len(results)} links for '{query}'.")
                else:
                    print(f"Bing HTML search failed for '{query}' with status: {response.status}")
        except asyncio.TimeoutError:
            print(f"Bing HTML search timed out for query: {query}")
        except Exception as e:
            print(f"Bing HTML search error for '{query}': {e}")
        return results[:max_results]

    async def _search_startpage(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """Startpage Suche (nutzt Google-Ergebnisse). Returns list of dicts {title, url, snippet}."""
        # --- This method was mostly fine, added timeout, ensure session ---
        results: List[Dict[str, str]] = []
        # Startpage POST request details might change. This is based on common patterns.
        # It's often better to use their official /search endpoint if available and stable.
        # For this example, using provided GET structure.
        search_url = "https://www.startpage.com/sp/search" # The original GET URL
        # Parameters might need to be specific to what startpage.com expects for web results vs image/video.
        # 'cat=web' 'cmd=process_search' 'language=english' 'enginecount=1' ' όπου=ENG' 'ใช่=คุณ' are sometimes seen.
        # Sticking to simple query for now.
        params = {"query": quote(query), "lui": "english"} # Ensure query is URL encoded
        
        headers = { # Startpage can be sensitive to User-Agent.
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
        try:
            if not self.session or self.session.closed: await self.initialize()
            async with self.session.get(search_url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=20)) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    # Startpage class names can change. These are examples.
                    # Look for result blocks, e.g., class "w-gl__result" or "result-link".
                    # The original prompt used 'div', class_='w-gl__result'
                    
                    # Common structure: section class="w-gl" -> div class="w-gl__result"
                    result_container = soup.find('section', class_='w-gl')
                    if not result_container : result_container = soup # fallback to whole soup
                        
                    for res_block in result_container.find_all('div', class_=['w-gl__result', 'search-result'], limit=max_results):
                        title_tag = res_block.find('a', class_=['w-gl__result-title', 'result-link__title'])
                        snippet_tag = res_block.find('p', class_=['w-gl__description', 'result-link__snippet'])
                        
                        if title_tag and title_tag.get('href'):
                            results.append({
                                "title": title_tag.get_text(strip=True),
                                "url": title_tag['href'],
                                "snippet": snippet_tag.get_text(strip=True) if snippet_tag else ""
                            })
                    print(f"Startpage search found {len(results)} links for '{query}'.")
                else:
                    print(f"Startpage search failed for '{query}' with status: {response.status} {await response.text()}")
        except asyncio.TimeoutError:
            print(f"Startpage search timed out for query: {query}")
        except Exception as e:
            print(f"Startpage search error for '{query}': {e}")
        return results[:max_results]


    async def _extract_web_contents(self,
                                    search_results: List[Dict[str, str]], # Expects dicts with 'url', 'title', 'snippet', 'engine'
                                    search_query_obj: SearchQuery) -> List[WebContent]: # --- ADDED: search_query_obj for context ---
        """Extrahiert Content aus Suchergebnissen."""
        # --- MODIFIED: More robust batching and error handling ---
        contents: List[WebContent] = []
        if not search_results: return contents

        # Limit extraction to max_pages_per_domain
        domain_counts: Dict[str, int] = {}
        urls_to_process: List[Dict[str,str]] = []
        for res_dict in search_results:
            url = res_dict.get("url")
            if not url: continue
            try:
                domain = urlparse(url).netloc.lower()
                if domain_counts.get(domain, 0) < self.config.max_pages_per_domain:
                    urls_to_process.append(res_dict)
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                else:
                    print(f"ℹ️ Skipping URL due to max_pages_per_domain for {domain}: {url}")
            except Exception as e_parse:
                print(f"⚠️ Error parsing URL for domain limiting: {url} ({e_parse})")
                urls_to_process.append(res_dict) # Process if parsing fails

        if not urls_to_process: return contents
        
        print(f"Attempting to extract content from {len(urls_to_process)} URLs...")
        self.successful_extractions = 0 # Reset counters for this research call
        self.failed_extractions = 0

        # Use a semaphore to limit concurrency for web requests more gracefully
        # Batching is still good for overall structure and rate limiting between batches.
        semaphore = asyncio.Semaphore(5) # Limit to 5 concurrent extractions

        async def fetch_and_extract_with_semaphore(res_dict: Dict[str,str]):
            async with semaphore:
                # --- Pass search_query_obj.user_id, not search_query_obj itself ---
                return await self._extract_single_content(res_dict, search_query_obj.user_id)


        tasks = [fetch_and_extract_with_semaphore(res_dict) for res_dict in urls_to_process]
        
        # No explicit batching loop here, rely on semaphore for concurrency control of _extract_single_content calls.
        # Rate limiting is per domain inside _extract_single_content or globally before calls.
        # The original outer batching loop with sleep is still a good idea for politeness across different domains.
        # Let's reintroduce that outer batching for the actual calls.

        batch_size = 5 # Number of tasks to launch before a potential pause
        for i in range(0, len(urls_to_process), batch_size):
            current_batch_search_results = urls_to_process[i : i + batch_size]
            batch_tasks = [fetch_and_extract_with_semaphore(res_dict) for res_dict in current_batch_search_results]
            
            task_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            for single_result in task_results:
                if isinstance(single_result, WebContent):
                    contents.append(single_result)
                    self.successful_extractions +=1
                elif isinstance(single_result, Exception):
                    # Error already printed in _extract_single_content or its sub-methods
                    self.failed_extractions +=1
                # None results are also failures, counted if _extract_single_content returns None explicitly
                elif single_result is None:
                     self.failed_extractions +=1
            
            if i + batch_size < len(urls_to_process) and self.config.rate_limit_delay > 0:
                print(f"--- Processed batch, pausing for {self.config.rate_limit_delay}s ---")
                await asyncio.sleep(self.config.rate_limit_delay)
                
        return contents


    async def _extract_single_content(self,
                                      search_result_dict: Dict[str, str], # --- MODIFIED: Takes the dict ---
                                      user_id: str) -> Optional[WebContent]:
        """Extrahiert Content von einer einzelnen URL, trying multiple methods."""
        url = search_result_dict.get("url")
        if not url: return None

        # --- cache_key should be just based on URL for content fetching ---
        # content_hash is for the extracted text, not for caching the WebContent object before extraction.
        url_cache_key = hashlib.md5(url.encode('utf-8')).hexdigest()
        if url_cache_key in self.content_cache:
            print(f"CACHE HIT for URL: {url}")
            return self.content_cache[url_cache_key]

        print(f"🚀 Extracting content from: {url}")
        extracted_wc: Optional[WebContent] = None
        extraction_method_used = "unknown"

        # Ensure session is active
        if not self.session or self.session.closed: await self.initialize()

        # --- MODIFIED: Try extractors in order ---
        # 1. Feedparser (if URL is a feed)
        try {
            extracted_wc = await self._extract_with_feedparser(url, search_result_dict)
            if extracted_wc: extraction_method_used = "feedparser"
        } except Exception as e_feed: {
             print(f"ℹ️ Feedparser attempt failed for {url}: {e_feed}")
        }


        # 2. Newspaper3k (good for articles)
        if not extracted_wc or len(extracted_wc.content) < 100: # If feed failed or gave minimal content
            try {
                temp_wc = await self._extract_with_newspaper(url)
                if temp_wc and (not extracted_wc or len(temp_wc.content) > len(extracted_wc.content if extracted_wc else "")):
                    extracted_wc = temp_wc
                    extraction_method_used = "newspaper"
            } except Exception as e_news: {
                print(f"ℹ️ Newspaper3k extraction failed for {url}: {e_news}")
            }

        # 3. BeautifulSoup (general HTML scraping)
        if not extracted_wc or len(extracted_wc.content) < 200: # Threshold for meaningful content
             try {
                temp_wc = await self._extract_with_bs4(url, search_result_dict) # Pass original search result for title fallback
                if temp_wc and (not extracted_wc or len(temp_wc.content) > len(extracted_wc.content if extracted_wc else "")):
                    extracted_wc = temp_wc
                    extraction_method_used = "beautifulsoup"
             } except Exception as e_bs4: {
                  print(f"ℹ️ BeautifulSoup extraction failed for {url}: {e_bs4}")
             }


        # 4. Selenium (JS-heavy sites, if enabled and other methods failed)
        if (not extracted_wc or len(extracted_wc.content) < 200) and self.config.enable_javascript and self._selenium_driver:
            print(f"ℹ️ Trying Selenium for {url} as other methods yielded little content.")
            try {
                temp_wc = await self._extract_with_selenium(url, search_result_dict)
                if temp_wc and (not extracted_wc or len(temp_wc.content) > len(extracted_wc.content if extracted_wc else "")):
                    extracted_wc = temp_wc
                    extraction_method_used = "selenium"
            } except Exception as e_sel: {
                 print(f"ℹ️ Selenium extraction failed for {url}: {e_sel}")
            }


        if extracted_wc and len(extracted_wc.content) >= 100: # Final check for meaningful content length
            print(f"✅ Successfully extracted content from {url} using {extraction_method_used} (Length: {len(extracted_wc.content)})")
            extracted_wc.content_hash = hashlib.md5(extracted_wc.content.encode('utf-8')).hexdigest()
            if self.config.save_raw_content and not extracted_wc.raw_html: # Save raw HTML if not already populated by extractor
                 # This would require fetching the page again if not saved by the extractor.
                 # For simplicity, we assume extractors might populate it or it's fetched by bs4/selenium.
                 pass

            extracted_wc.source_engine = search_result_dict.get("engine", "unknown") # Store which engine found this
            
            # Auto-summarize and embed if enabled and not already done by a document processor step
            if self.config.auto_summarize and not extracted_wc.summary and self.llm_handler:
                extracted_wc.summary = await self._generate_content_summary(extracted_wc.content)
            
            if self.embedding_engine and not extracted_wc.embedding:
                text_for_embedding = f"Title: {extracted_wc.title}\nSummary: {extracted_wc.summary}\nContent Snippet: {extracted_wc.content[:1000]}"
                embedding_result = await self.embedding_engine.embed([text_for_embedding])
                if embedding_result and embedding_result.embeddings:
                    raw_emb = embedding_result.embeddings[0]
                    extracted_wc.embedding = raw_emb.tolist() if hasattr(raw_emb, 'tolist') else raw_emb
            
            self.content_cache[url_cache_key] = extracted_wc # Cache the successfully extracted and processed content
            return extracted_wc
        else:
            print(f"⚠️ Failed to extract sufficient content from {url} after trying all methods.")
            return None


    async def _extract_with_feedparser(self, url: str, search_result_dict: Dict[str,str]) -> Optional[WebContent]: # --- ADDED ---
        """Content Extraction from RSS/Atom feeds."""
        print(f"Attempting feed parsing for {url}...")
        loop = asyncio.get_event_loop()
        try:
            # feedparser.parse is blocking
            feed_data = await loop.run_in_executor(None, feedparser.parse, url)

            if not feed_data or not feed_data.feed or not feed_data.entries:
                # print(f"Not a valid feed or no entries: {url}")
                return None

            # Successfully parsed a feed. We need to decide what to return.
            # For _extract_single_content, we typically expect one page's content.
            # Option 1: Return info about the feed itself (title, description).
            # Option 2: Return the content of the *first* or *most relevant* entry.
            # Let's go with Option 2, trying to find an entry matching the original search result title if possible.
            
            target_entry = None
            original_search_title = search_result_dict.get("title", "").lower()

            if original_search_title:
                for entry in feed_data.entries:
                    if entry.get("title", "").lower() == original_search_title:
                        target_entry = entry
                        break
            
            if not target_entry and feed_data.entries: # Fallback to first entry
                target_entry = feed_data.entries[0]

            if not target_entry:
                return None # No suitable entry found

            print(f"Feed entry found: '{target_entry.get('title', 'Untitled')}' from {url}")

            content_text = ""
            if 'content' in target_entry and target_entry.content: # List of content dicts
                content_text = BeautifulSoup(target_entry.content[0].value, 'html.parser').get_text(separator=' ', strip=True)
            elif 'summary' in target_entry:
                content_text = BeautifulSoup(target_entry.summary, 'html.parser').get_text(separator=' ', strip=True)
            elif 'description' in target_entry: # Common in older RSS
                content_text = BeautifulSoup(target_entry.description, 'html.parser').get_text(separator=' ', strip=True)


            if not content_text or len(content_text) < 100: return None # Skip if too little content from entry

            entry_url = target_entry.get('link', url) # Use entry link if available, else feed URL
            entry_title = target_entry.get('title', search_result_dict.get("title", "Untitled Feed Entry"))
            
            publish_dt = None
            if 'published_parsed' in target_entry and target_entry.published_parsed:
                try:
                    publish_dt = datetime.fromtimestamp(time.mktime(target_entry.published_parsed))
                except Exception: pass
            elif 'updated_parsed' in target_entry and target_entry.updated_parsed:
                 try:
                    publish_dt = datetime.fromtimestamp(time.mktime(target_entry.updated_parsed))
                 except Exception: pass


            return WebContent(
                url=entry_url, title=entry_title, content=content_text[:self.config.max_content_length],
                summary=None, # Generate later
                author=target_entry.get('author') or (feed_data.feed.get('author') if feed_data.feed else None),
                publish_date=publish_dt,
                domain=urlparse(entry_url).netloc, language=target_entry.get('language') or feed_data.feed.get('language'),
                word_count=len(content_text.split()), links=[e.get('link') for e in feed_data.entries if e.get('link')], # Links to other entries
                images=[], # Feed entries might not list images directly here
                metadata={"extraction_method": "feedparser", "feed_title": feed_data.feed.get('title')},
                extraction_time=datetime.utcnow(), content_hash="" # Hash later
            )

        except Exception as e:
            print(f"Feedparser extraction failed for {url}: {e}")
            return None


    async def _extract_with_newspaper(self, url: str) -> Optional[WebContent]:
        """Content Extraction mit Newspaper3k, run in executor."""
        # --- MODIFIED: Made blocking calls run in executor ---
        print(f"Attempting Newspaper3k extraction for {url}...")
        loop = asyncio.get_event_loop()
        try:
            article = newspaper.Article(url, fetch_images=False, request_timeout=self.config.request_timeout//2) # Disable image fetching initially for speed
            # These are blocking I/O calls
            await loop.run_in_executor(None, article.download)
            if not article.is_downloaded:
                 print(f"Newspaper failed to download: {url}")
                 return None
            await loop.run_in_executor(None, article.parse)

            if not article.text or len(article.text) < 100: # Check for minimal content
                # print(f"Newspaper extracted very little text from: {url}")
                return None
            
            content_text = article.text[:self.config.max_content_length] # Truncate if too long

            metadata = {
                "extraction_method": "newspaper",
                "top_image_url": article.top_image, # URL string
                "movies": article.movies, # List of video URLs
                "tags": list(article.tags) if article.tags else (article.keywords or []), # Combined tags/keywords
                "meta_description": article.meta_description,
                "meta_language": article.meta_lang,
                "canonical_link": article.canonical_link
            }

            return WebContent(
                url=url, title=article.title or "Untitled", content=content_text,
                summary=None, # Will be generated later if enabled
                author=", ".join(article.authors) if article.authors else None,
                publish_date=article.publish_date, # Already a datetime object
                domain=urlparse(url).netloc, language=article.meta_lang or None, # Use None if not detected
                word_count=len(content_text.split()),
                links=list(article.html_links()) if self.config.extract_links else [], # Extract links if needed
                images=[article.top_image] if article.top_image else [], # Newspaper mainly gets top_image
                metadata=metadata, extraction_time=datetime.utcnow(), content_hash="" # Hash later
            )
        except newspaper.article.ArticleException as e_art: # Specific newspaper exception
            print(f"Newspaper ArticleException for {url}: {e_art} (Likely not an article or access issue)")
            return None
        except Exception as e:
            print(f"Newspaper3k general extraction error for {url}: {e}")
            return None


    async def _extract_with_bs4(self,
                                url: str,
                                search_result_dict: Optional[Dict[str, str]] = None) -> Optional[WebContent]: # --- MODIFIED: search_result_dict is Optional ---
        """Content Extraction mit BeautifulSoup."""
        # --- This method was largely okay, added raw_html saving, ensure session ---
        print(f"Attempting BeautifulSoup extraction for {url}...")
        try:
            if not self.session or self.session.closed: await self.initialize()
            async with self.session.get(url, allow_redirects=self.config.follow_redirects, timeout=self.config.request_timeout) as response:
                if response.status != 200:
                    print(f"BS4: HTTP error {response.status} for {url}")
                    return None
                
                try:
                    html_content = await response.text(encoding='utf-8', errors='replace') # Specify encoding
                except (UnicodeDecodeError, LookupError) as enc_err: # Handle encoding issues
                    print(f"BS4: Encoding error for {url}: {enc_err}. Trying with detected encoding or ISO-8859-1.")
                    try:
                        detected_encoding = response.charset or 'iso-8859-1' # Fallback encoding
                        html_content = await response.text(encoding=detected_encoding, errors='replace')
                    except Exception as e_retry_enc:
                         print(f"BS4: Retried encoding also failed for {url}: {e_retry_enc}")
                         return None

                if not html_content: return None

                raw_html_to_store = html_content if self.config.save_raw_content else None
                
                if len(html_content) > self.config.max_content_length * 1.5: # Allow slightly larger raw HTML before parsing
                    print(f"BS4: Truncating very large HTML content for {url} before parsing.")
                    html_content = html_content[:int(self.config.max_content_length * 1.5)]

                soup = BeautifulSoup(html_content, 'lxml') # --- MODIFIED: Use lxml for speed if available, fallback to html.parser ---
                                                        # User needs to `pip install lxml`

                # Remove unwanted tags
                for unwanted_tag in soup(["script", "style", "nav", "footer", "aside", "form", "header", "iframe", "noscript", "select", "button", "input", "textarea"]):
                    unwanted_tag.decompose()
                
                title = soup.find('title')
                page_title = title.string.strip() if title and title.string else (search_result_dict.get("title") if search_result_dict else "Untitled")

                # More targeted content extraction
                main_content_area = soup.find(['main', 'article']) or \
                                    soup.find(attrs={'role': 'main'}) or \
                                    soup.find(id=['content', 'main-content', 'articleBody']) or \
                                    soup.find(class_=['content', 'post-content', 'entry-content', 'article-body', 'mainContent'])
                
                if main_content_area:
                    # Further clean inside main content: remove share buttons, related links divs, etc.
                    for el in main_content_area.find_all(class_=["share", "related-posts", "comments", "sidebar"]): el.decompose()
                    text_content = main_content_area.get_text(separator='\n', strip=True) # Use newline as separator then clean
                else: # Fallback to body if no main content area found
                    body_tag = soup.find('body')
                    text_content = body_tag.get_text(separator='\n', strip=True) if body_tag else soup.get_text(separator='\n', strip=True)

                # Clean multiple newlines and leading/trailing whitespace
                text_content = re.sub(r'\n\s*\n', '\n', text_content).strip()
                text_content = text_content[:self.config.max_content_length] # Truncate

                if len(text_content) < 100: # Check for minimal content length
                    # print(f"BS4 extracted very little text from: {url}")
                    return None

                # Links and Images (simplified)
                links_extracted = []
                if self.config.extract_links:
                    for a_tag in (main_content_area or soup).find_all('a', href=True, limit=20):
                        href = a_tag['href']
                        if href and not href.startswith(('#', 'javascript:')):
                            full_url = urljoin(url, href) # Resolve relative URLs
                            if urlparse(full_url).netloc : links_extracted.append(full_url) # Ensure it's a valid, absolute URL

                images_extracted = []
                for img_tag in (main_content_area or soup).find_all('img', src=True, limit=10):
                    src = img_tag.get('src','')
                    if src and not src.startswith('data:'): # Exclude data URIs
                        full_img_url = urljoin(url, src)
                        if urlparse(full_img_url).netloc : images_extracted.append(full_img_url)


                # Basic metadata from meta tags
                meta_desc_tag = soup.find('meta', attrs={'name': re.compile(r'description', re.I)})
                meta_keywords_tag = soup.find('meta', attrs={'name': re.compile(r'keywords', re.I)})
                meta_author_tag = soup.find('meta', attrs={'name': re.compile(r'author', re.I)})
                
                publish_date_obj = None # TODO: Add date parsing from meta tags or structured data
                
                lang_tag = soup.find('html', lang=True)
                page_lang = lang_tag['lang'] if lang_tag else None


                return WebContent(
                    url=url, title=page_title, content=text_content,
                    summary=None, # Generate later
                    author=meta_author_tag['content'].strip() if meta_author_tag and meta_author_tag.get('content') else None,
                    publish_date=publish_date_obj,
                    domain=urlparse(url).netloc, language=page_lang,
                    word_count=len(text_content.split()),
                    links=list(set(links_extracted)), images=list(set(images_extracted)),
                    metadata={
                        "extraction_method": "beautifulsoup",
                        "meta_description": meta_desc_tag['content'].strip() if meta_desc_tag and meta_desc_tag.get('content') else None,
                        "meta_keywords": [k.strip() for k in meta_keywords_tag['content'].split(',')] if meta_keywords_tag and meta_keywords_tag.get('content') else []
                    },
                    extraction_time=datetime.utcnow(), content_hash="", # Hash later
                    raw_html=raw_html_to_store
                )
        except UnicodeDecodeError as ude: # Catch specific error
             print(f"BeautifulSoup extraction UnicodeDecodeError for {url}: {ude}")
             return None
        except Exception as e:
            print(f"BeautifulSoup extraction error for {url}: {e}")
            return None


    async def _extract_with_selenium(self,
                                     url: str,
                                     search_result_dict: Optional[Dict[str, str]] = None) -> Optional[WebContent]: # --- MODIFIED: search_result_dict Optional ---
        """Content Extraction mit Selenium, run in executor, with better content selection."""
        if not self.config.enable_javascript or not self._selenium_driver:
            # print(f"Selenium is disabled or driver not initialized. Skipping for {url}")
            return None
        
        print(f"Attempting Selenium extraction for {url}...")
        loop = asyncio.get_event_loop()
        driver = self._selenium_driver # Use the shared driver instance

        try:
            # --- MODIFIED: Selenium operations in executor ---
            def get_page_content(driver_instance, page_url, wait_timeout):
                driver_instance.get(page_url)
                # Wait for a common element like 'body' or a specific main content area if known
                WebDriverWait(driver_instance, wait_timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                # --- ADDED: Try to scroll to load dynamic content ---
                driver_instance.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
                time.sleep(1) # Brief pause for any JS loading after scroll
                driver_instance.execute_script("window.scrollTo(0, 2*document.body.scrollHeight/3);")
                time.sleep(1)
                driver_instance.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2) # Longer pause after full scroll

                page_title = driver_instance.title
                page_source_html = driver_instance.page_source # Get full HTML after JS execution
                return page_title, page_source_html

            page_title, html_content = await loop.run_in_executor(None, get_page_content, driver, url, self.config.selenium_wait_timeout)
            
            if not html_content:
                print(f"Selenium: No page source obtained for {url}")
                return None

            # --- MODIFIED: Use BeautifulSoup to parse Selenium's output for better content extraction ---
            # This reuses the more sophisticated BS4 parsing logic on dynamic content.
            soup = BeautifulSoup(html_content, 'lxml') # Use lxml for consistency

            for unwanted_tag in soup(["script", "style", "nav", "footer", "aside", "form", "header", "iframe", "noscript"]):
                unwanted_tag.decompose()
            
            # Determine title (Selenium's title or fallback)
            final_title = page_title or (search_result_dict.get("title") if search_result_dict else "Untitled")

            main_content_area = soup.find(['main', 'article']) or \
                                soup.find(attrs={'role': 'main'}) or \
                                soup.find(id=['content', 'main-content', 'articleBody']) or \
                                soup.find(class_=['content', 'post-content', 'entry-content', 'article-body', 'mainContent'])
            
            text_content = ""
            if main_content_area:
                text_content = main_content_area.get_text(separator='\n', strip=True)
            else:
                body_tag = soup.find('body')
                text_content = body_tag.get_text(separator='\n', strip=True) if body_tag else soup.get_text(separator='\n', strip=True)

            text_content = re.sub(r'\n\s*\n', '\n', text_content).strip()
            text_content = text_content[:self.config.max_content_length]

            if len(text_content) < 100: # Check for minimal content
                print(f"Selenium extracted very little text from: {url}")
                return None
            
            # Links and Images (simplified, could be enhanced with Selenium selectors for visible elements)
            links_extracted = []
            if self.config.extract_links:
                for a_tag in soup.find_all('a', href=True, limit=20): # Parse from soup
                    href = a_tag['href']
                    if href and not href.startswith(('#', 'javascript:')):
                        full_url = urljoin(url, href)
                        if urlparse(full_url).netloc: links_extracted.append(full_url)
            
            images_extracted = [] # Could use Selenium to find visible images if needed

            return WebContent(
                url=url, title=final_title, content=text_content,
                summary=None, author=None, publish_date=None, # These are harder to get reliably with basic Selenium text dump
                domain=urlparse(url).netloc, language=None, # Language from Selenium is harder
                word_count=len(text_content.split()),
                links=list(set(links_extracted)), images=images_extracted,
                metadata={"extraction_method": "selenium"},
                extraction_time=datetime.utcnow(), content_hash="", # Hash later
                raw_html=html_content if self.config.save_raw_content else None
            )

        except TimeoutError: # This is Python's built-in TimeoutError, Selenium has its own
            print(f"Selenium operation timed out for {url} (Python TimeoutError).")
            return None
        except webdriver.common.exceptions.TimeoutException: # Selenium specific timeout
            print(f"Selenium explicit wait timed out for {url}.")
            return None
        except Exception as e:
            print(f"Selenium extraction error for {url}: {type(e).__name__} - {e}")
            return None
        # Note: Selenium driver `quit()` is handled in the main `close()` method.

    async def _process_contents(self,
                                web_contents: List[WebContent],
                                user_id: str) -> List[WebContent]: # --- MODIFIED: This method might be redundant if processing happens in _extract_single_content ---
        """
        Verarbeitet Web Contents mit Document Processor.
        This is mainly for chunking, advanced summarization, or if embeddings weren't generated during extraction.
        """
        if not self.document_processor:
            # print("Document processor not available, skipping this processing step.")
            return web_contents # Return as is if no processor

        processed_list: List[WebContent] = []
        for wc_item in web_contents:
            if not wc_item.content: # Skip if no content
                processed_list.append(wc_item)
                continue
            try:
                # If summary/embedding already done, document_processor might skip or refine them.
                # This assumes document_processor can handle WebContent-like structures or just text.
                # For now, let's assume we process the main content for chunking/metadata.
                
                # We pass text and existing metadata. DocumentProcessor typically expects a path or raw text.
                # Let's use 'process_text_directly' as in the original prompt.
                proc_metadata = {
                    "source_url": wc_item.url, "source_domain": wc_item.domain,
                    "original_title": wc_item.title,
                    "original_author": wc_item.author,
                    "original_publish_date": wc_item.publish_date.isoformat() if wc_item.publish_date else None,
                    "initial_extraction_method": wc_item.metadata.get("extraction_method", "unknown")
                }
                
                # DocumentProcessor might return multiple chunks or a single processed document object.
                # Assuming `process_text_directly` returns an object similar to `ProcessedDocument`
                # (as defined in placeholder if actual tools.document_processor is missing)
                # For this example, let's assume it primarily refines the summary and ensures embedding.
                
                processed_doc_result = await self.document_processor.process_text_directly(
                    text=wc_item.content,
                    title=wc_item.title,
                    user_id=user_id, # Pass user_id
                    metadata=proc_metadata
                )

                # Update WebContent with results from DocumentProcessor
                # This depends heavily on what `processed_doc_result` structure is.
                # Assuming it has attributes like 'summary', 'chunks' (where chunks have embeddings).
                
                if hasattr(processed_doc_result, 'summary') and processed_doc_result.summary:
                    wc_item.summary = processed_doc_result.summary
                
                if not wc_item.embedding and hasattr(processed_doc_result, 'chunks') and \
                   processed_doc_result.chunks and hasattr(processed_doc_result.chunks[0], 'embedding') \
                   and processed_doc_result.chunks[0].embedding:
                    # Use embedding of the first chunk as representative for the document, or average them.
                    emb = processed_doc_result.chunks[0].embedding
                    wc_item.embedding = emb.tolist() if hasattr(emb, 'tolist') else emb
                
                # DocumentProcessor might also update other fields like cleaned content, keywords etc.
                # if hasattr(processed_doc_result, 'cleaned_text') and processed_doc_result.cleaned_text:
                #    wc_item.content = processed_doc_result.cleaned_text
                # if hasattr(processed_doc_result, 'keywords') and processed_doc_result.keywords:
                #    wc_item.metadata['processed_keywords'] = processed_doc_result.keywords

                processed_list.append(wc_item)

            except Exception as e_proc:
                print(f"⚠️ Error during document processing for {wc_item.url}: {e_proc}")
                processed_list.append(wc_item) # Append original if processing fails
        return processed_list


    async def _generate_content_summary(self, content_text: str) -> Optional[str]: # --- Parameter renamed ---
        """Generiert automatische Zusammenfassung für Content via LLM."""
        # --- This method was mostly fine, added check for LLM handler ---
        if not self.llm_handler or not content_text or len(content_text.strip()) < 200 : # Min length for summary
            return None
        
        # Truncate content for summarization prompt to fit context window and save tokens
        # Taking a snippet from start, middle, and end might be better for very long texts.
        # For now, simple truncation.
        content_snippet_for_summary = content_text[:min(len(content_text), 4000)] # Max ~4k chars for summary context

        summary_prompt_template = f"""
        Based on the following text, please provide a concise, factual summary in 2-4 sentences.
        Focus on the main topic and key information presented.

        Text to Summarize:
        ---
        {content_snippet_for_summary}
        ---
        Concise Summary (2-4 sentences):
        """
        try:
            llm_req = LLMRequest(
                prompt=summary_prompt_template,
                user_id="research_summarizer_system", # System user for this task
                max_tokens=200, # Max tokens for the summary itself
                temperature=0.2, # Lower temperature for factual summary
                # Add other params like top_p, stop_sequences if needed by your LLM handler
            )
            llm_response = await self.llm_handler.generate_single(llm_req)

            if llm_response and llm_response.response:
                summary = llm_response.response.strip()
                # Basic validation of summary
                if len(summary) > 20 and not summary.lower().startswith(("i cannot", "i am unable", "sorry")):
                    return summary
            return None
        except Exception as e_sum:
            print(f"⚠️ LLM summary generation failed: {e_sum}")
            return None


    async def _rerank_contents(self,
                               query: str,
                               contents: List[WebContent]) -> List[WebContent]:
        """Re-rankt Content nach Relevanz mit MLXRerankEngine."""
        # --- This method was mostly fine, using more descriptive text for reranking ---
        if not self.rerank_engine or not contents or len(contents) <=1:
            return contents # No reranking needed or possible

        print(f"Re-ranking {len(contents)} items for query: '{query}'")
        try:
            # Prepare candidates for the reranker
            rerank_candidates = []
            for idx, wc_item in enumerate(contents):
                # Text for reranking: title + summary, or title + start of content
                text_for_rerank = f"Title: {wc_item.title}\n"
                if wc_item.summary:
                    text_for_rerank += f"Summary: {wc_item.summary}\n"
                else: # Use snippet of content if no summary
                    text_for_rerank += f"Content Snippet: {wc_item.content[:1000]}\n" # Use a good chunk
                
                rerank_candidates.append({
                    "id": str(idx), # Use index as ID to map back to original WebContent objects
                    "text": text_for_rerank, # --- MODIFIED: use text key, MLXRerankEngine might expect this ---
                    # "content": text_for_rerank, # Or 'content' depending on reranker's expected input
                    "metadata": {"original_url": wc_item.url, "title": wc_item.title}, # Minimal metadata for reranker
                    # "score": 0.0 # Initial score if needed by reranker, often not
                })

            # Perform reranking
            # Assuming rerank method of MLXRerankEngine is defined and works
            rerank_result_obj = await self.rerank_engine.rerank(
                query=query,
                candidates=rerank_candidates, # List of dicts
                top_k=min(self.config.max_search_results, len(rerank_candidates)) # Rerank up to max configured results
            )

            # Map reranked results (which are dicts with 'id' and 'score') back to WebContent objects
            reranked_webcontents: List[WebContent] = []
            if rerank_result_obj and rerank_result_obj.candidates:
                for reranked_item_dict in rerank_result_obj.candidates:
                    original_idx = int(reranked_item_dict['id']) # Get back the original index
                    if 0 <= original_idx < len(contents):
                        # Optionally, store the rerank score in WebContent.metadata
                        contents[original_idx].metadata['rerank_score'] = reranked_item_dict.get('score', 0.0)
                        reranked_webcontents.append(contents[original_idx])
                return reranked_webcontents
            else: # Reranking failed or returned no candidates
                print("⚠️ Reranking did not return candidates, returning original order.")
                return contents

        except Exception as e_rerank:
            print(f"⚠️ Reranking process error: {e_rerank}")
            return contents # Fallback to original list if reranking fails


    async def _synthesize_answer(self,
                                 search_query: SearchQuery,
                                 top_contents: List[WebContent]) -> Dict[str, Any]: # --- RENAMED contents to top_contents ---
        """Synthetisiert Antwort aus Top-Contents mit RAG via LLM."""
        # --- This method was mostly fine, refining context and prompt ---
        if not top_contents:
            return {"answer": "I could not find enough information to answer your query.", "sources": [], "confidence": 0.1}
        if not self.llm_handler:
            return {"answer": "LLM service is not available to synthesize an answer.", "sources": [], "confidence": 0.0}

        print(f"Synthesizing answer for '{search_query.query}' using {len(top_contents)} sources.")
        
        context_parts_for_llm: List[str] = []
        sources_for_citation: List[Dict[str,str]] = []
        max_context_chars = 6000 # Limit total characters passed to LLM from sources
        current_chars = 0

        for idx, wc_item in enumerate(top_contents[:5]): # Use up to top 5 for synthesis context
            source_ref = f"[Source {idx+1}: {wc_item.domain} - \"{wc_item.title}\"]"
            
            text_to_include = wc_item.summary if wc_item.summary and len(wc_item.summary) > 50 else wc_item.content
            # Truncate individual source text to avoid one source dominating the context window
            max_chars_per_source = max_context_chars // min(len(top_contents), 5) # Distribute budget
            text_snippet = text_to_include[:min(len(text_to_include), max_chars_per_source)]

            if current_chars + len(text_snippet) + len(source_ref) > max_context_chars:
                break # Stop if adding next source exceeds budget

            context_parts_for_llm.append(f"{source_ref}\n{text_snippet}\n---")
            sources_for_citation.append({"title": wc_item.title, "url": wc_item.url, "domain": wc_item.domain, "ref": f"[Source {idx+1}]"})
            current_chars += len(text_snippet) + len(source_ref)

        if not context_parts_for_llm:
             return {"answer": "Although sources were found, they could not be processed for answer synthesis.", "sources": [], "confidence": 0.15}

        full_context_str = "\n\n".join(context_parts_for_llm)

        synthesis_prompt = f"""
        You are a helpful research assistant. Your task is to answer the user's question based *only* on the provided context from web sources.
        Do not use any prior knowledge. If the context does not provide an answer, state that clearly.

        User's Question: "{search_query.query}"
        {'User-provided Context (if any): "' + search_query.context + '"' if search_query.context else ''}

        Provided Web Sources Context:
        ```
        {full_context_str}
        ```

        Instructions for your answer:
        1. Answer the user's question comprehensively using information from the provided web sources.
        2. When you use information from a source, cite it using its reference (e.g., [Source 1], [Source 2]).
        3. If the sources offer conflicting information, present the different viewpoints and cite them.
        4. If the sources do not contain enough information to fully answer the question, explicitly state what information is missing or cannot be determined from the context.
        5. Structure your answer clearly. Use bullet points or numbered lists if appropriate for readability.
        6. Maintain a neutral, objective, and factual tone. Avoid personal opinions or speculation.
        7. Do NOT invent information or answer from outside the provided context.

        Synthesized Answer:
        """
        try:
            llm_req = LLMRequest(
                prompt=synthesis_prompt,
                user_id=search_query.user_id, # For tracking/logging if LLM handler supports it
                max_tokens=1024, # Allow longer, more comprehensive answers
                temperature=0.1, # Low temperature for factual, less creative answers
                # stop_sequences=["\n\n---"], # Optional: if LLM tends to add extra unwanted sections
            )
            llm_response = await self.llm_handler.generate_single(llm_req)

            final_answer = llm_response.response.strip() if llm_response and llm_response.response else \
                           "The language model could not generate an answer based on the provided sources."
            
            # Basic check if answer actually used sources (heuristic)
            confidence = self._calculate_confidence_score(top_contents, final_answer) # Pass top_contents used for synthesis
            if not any(f"[Source {i+1}]" in final_answer for i in range(len(sources_for_citation))):
                if confidence > 0.3: confidence -= 0.2 # Reduce confidence if no citations are found

            return {"answer": final_answer, "sources": sources_for_citation, "confidence": max(0.05, confidence)}

        except Exception as e_synth:
            print(f"⚠️ LLM answer synthesis error: {e_synth}")
            return {"answer": f"Error during answer synthesis: {str(e_synth)}", "sources": sources_for_citation, "confidence": 0.0}


    def _calculate_confidence_score(self,
                                    used_contents: List[WebContent], # --- RENAMED contents to used_contents ---
                                    answer_text: str) -> float: # --- RENAMED answer to answer_text ---
        """Berechnet Confidence Score für die Antwort. More nuanced."""
        if not used_contents or not answer_text or answer_text.lower().startswith("i could not find"):
            return 0.1 # Low confidence if no sources or empty answer

        score = 0.0
        max_score_possible = 0.0

        # 1. Number of diverse sources used (max 0.3 points)
        num_sources = len(used_contents)
        unique_domains = len(set(c.domain for c in used_contents))
        score += min(num_sources / 3.0, 1.0) * 0.15  # Up to 3 sources contribute significantly
        score += min(unique_domains / 2.0, 1.0) * 0.15 # Up to 2 unique domains add to confidence
        max_score_possible += 0.3

        # 2. Quality & recency of sources (max 0.3 points)
        #    - Average word count (up to 0.1)
        #    - Presence of author/publish date (up to 0.1)
        #    - Recency (if dates available) (up to 0.1)
        total_word_count = sum(c.word_count for c in used_contents)
        avg_word_count_score = min( (total_word_count / num_sources if num_sources else 0) / 1000.0, 1.0) * 0.1
        score += avg_word_count_score
        max_score_possible += 0.1

        num_with_author_or_date = sum(1 for c in used_contents if c.author or c.publish_date)
        author_date_score = (num_with_author_or_date / num_sources if num_sources else 0) * 0.1
        score += author_date_score
        max_score_possible += 0.1
        
        # Recency scoring (e.g. last 1-2 years full points, older less)
        # This needs consistent date parsing. Assuming it's done.
        # For now, a simpler check if any content has a publish date.
        if any(c.publish_date for c in used_contents):
            # A more complex recency score could be added here.
            # For now, just +0.05 if any date exists.
            score += 0.05
        max_score_possible += 0.1 # Max possible for recency even if simple now.


        # 3. Answer characteristics (max 0.2 points)
        #    - Length of answer (up to 0.1)
        #    - Presence of citations (up to 0.1)
        answer_len_score = min(len(answer_text.split()) / 200.0, 1.0) * 0.1 # 200 words is a decent answer
        score += answer_len_score
        max_score_possible += 0.1

        if re.search(r"\[Source\s*\d+]", answer_text): # Check for "[Source N]"
            score += 0.1
        max_score_possible += 0.1

        # 4. Alignment/Consistency (Hard to measure without more semantic tools, placeholder for 0.2 points)
        #    - Could use embedding cosine similarity between query and answer, or query and source snippets.
        #    - For now, let's assume some base alignment if an answer is generated.
        if len(answer_text) > 50 and not answer_text.lower().startswith(("error", "sorry", "i cannot")):
            score += 0.1 # Base for generating a plausible answer
        max_score_possible += 0.2 # Max for alignment

        # Normalize score to be between 0.1 and 0.95
        normalized_score = (score / max_score_possible) if max_score_possible > 0 else 0.0
        return max(0.1, min(0.95, normalized_score)) # Ensure it's within a reasonable range


    async def _generate_follow_up_questions(self,
                                            original_query: str,
                                            answer_text: str) -> List[str]: # --- RENAMED answer to answer_text ---
        """Generiert Follow-up Fragen basierend auf Antwort via LLM."""
        # --- This method was mostly fine, ensuring LLM handler presence ---
        if not self.llm_handler or not answer_text or len(answer_text.strip()) < 50:
            return []
        
        # Take a snippet of the answer if it's too long for the prompt
        answer_snippet = answer_text[:min(len(answer_text), 1000)]

        followup_prompt_template = f"""
        Given the following research query and its synthesized answer, please generate 2-3 insightful follow-up questions a user might ask to delve deeper or explore related aspects.
        The follow-up questions should be distinct and encourage further investigation.

        Original Research Query:
        "{original_query}"

        Synthesized Answer Snippet:
        ---
        {answer_snippet}...
        ---

        Suggest 2-3 Follow-up Questions (each on a new line, starting with a hyphen):
        -
        """
        try:
            llm_req = LLMRequest(
                prompt=followup_prompt_template, user_id="research_followup_system",
                max_tokens=150, temperature=0.4, # Slightly more creative for questions
            )
            llm_response = await self.llm_handler.generate_single(llm_req)

            if llm_response and llm_response.response:
                raw_questions = llm_response.response.strip().split('\n')
                parsed_questions = []
                for q_line in raw_questions:
                    cleaned_q = q_line.strip()
                    if cleaned_q.startswith(("- ", "* ")): # Remove list markers
                        cleaned_q = cleaned_q[2:].strip()
                    if cleaned_q and len(cleaned_q) > 10 and cleaned_q.endswith("?"): # Basic validation
                        parsed_questions.append(cleaned_q)
                return parsed_questions[:3] # Return up to 3 valid questions
            return []
        except Exception as e_fuq:
            print(f"⚠️ LLM follow-up question generation error: {e_fuq}")
            return []


    async def _extract_related_topics(self, top_contents: List[WebContent]) -> List[str]: # --- RENAMED contents to top_contents ---
        """Extrahiert verwandte Themen aus Top-Contents. Can be LLM-based or keyword-based."""
        # --- This method was very basic, trying a slightly better keyword approach, or suggest LLM ---
        if not top_contents: return []

        # Option 1: Simple keyword/phrase extraction (improved from original)
        all_text_for_topics = ""
        for wc_item in top_contents[:3]: # Use top 3 for topic extraction
            all_text_for_topics += f"{wc_item.title}. {wc_item.summary or wc_item.content[:500]}. "
        
        if not all_text_for_topics.strip(): return []

        # Use LLM for more sophisticated topic extraction if available
        if self.llm_handler:
            try:
                topic_prompt = f"""
                Based on the following text aggregated from several web sources, identify 3-5 main related topics or key concepts.
                List each topic on a new line, starting with a hyphen.

                Aggregated Text Snippet:
                ---
                {all_text_for_topics[:2000]} 
                ---
                Related Topics/Key Concepts (3-5):
                - 
                """
                llm_req = LLMRequest(prompt=topic_prompt, user_id="research_topics_system", max_tokens=100, temperature=0.2)
                llm_response = await self.llm_handler.generate_single(llm_req)
                if llm_response and llm_response.response:
                    raw_topics = llm_response.response.strip().split('\n')
                    parsed_topics = []
                    for t_line in raw_topics:
                        cleaned_t = t_line.strip()
                        if cleaned_t.startswith(("- ", "* ")): cleaned_t = cleaned_t[2:].strip()
                        if cleaned_t and len(cleaned_t) > 2: parsed_topics.append(cleaned_t)
                    if parsed_topics: return parsed_topics[:5]
            except Exception as e_llm_topic:
                print(f"LLM-based topic extraction failed: {e_llm_topic}. Falling back to keyword method.")

        # Fallback to improved keyword extraction if LLM fails or not used
        try:
            # Simple regex for potential multi-word phrases (Capitalized Words or common nouns)
            # This is still very heuristic. A proper NLP library (spaCy, NLTK) would be better for robust keyword/NER.
            words_and_phrases = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b|\b[a-z]{4,}\b', all_text_for_topics)
            
            # Crude stopword list (extend as needed)
            stopwords_simple = {"the", "and", "is", "are", "of", "to", "in", "it", "that", "this", "for", "on", "with", 
                                "as", "was", "were", "be", "has", "have", "by", "at", "an", "or", "not", "from", "summary", "content"}
            
            term_freq: Dict[str, int] = {}
            for term in words_and_phrases:
                term_lower = term.lower()
                if len(term_lower) > 3 and term_lower not in stopwords_simple:
                    # Prefer original casing for capitalized terms if they are frequent
                    term_to_store = term if term[0].isupper() else term_lower
                    term_freq[term_to_store] = term_freq.get(term_to_store, 0) + 1
            
            # Sort by frequency, get top N
            # Filter out terms that are just substrings of other more frequent terms (simple version)
            sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
            
            final_topics: List[str] = []
            for term, _ in sorted_terms:
                is_substring_of_existing = False
                for existing_topic in final_topics:
                    if term.lower() in existing_topic.lower() and len(term) < len(existing_topic):
                        is_substring_of_existing = True
                        break
                if not is_substring_of_existing:
                    final_topics.append(term)
                if len(final_topics) >= 5: break
            return final_topics

        except Exception as e_topic:
            print(f"⚠️ Keyword-based topic extraction error: {e_topic}")
            return []


    async def save_research_to_brain(self,
                                     research_result: ResearchResult, # --- Parameter name updated ---
                                     brain_context_type: str = "research_finding") -> Dict[str, Any]: # --- Default type updated ---
        """Speichert Research Result im Brain System Format (für Kontext-Management)."""
        # --- This method was mostly fine, formatting and context name improved ---
        try:
            brain_content_parts = [
                f"# Research Finding: {research_result.query}",
                f"*Date: {research_result.timestamp.strftime('%Y-%m-%d %H:%M')} | User: {research_result.user_id}*",
                f"*Confidence: {research_result.confidence_score:.2f} | Duration: {research_result.research_time:.1f}s*",
                "---",
                "## Synthesized Answer:",
                research_result.synthesized_answer,
                "---"
            ]

            if research_result.sources_used:
                brain_content_parts.append("## Key Sources Consulted:")
                for i, src in enumerate(research_result.sources_used[:3]): # Show top 3-5 sources
                    brain_content_parts.append(f"{i+1}. **{src['title']}** ({src['domain']}) - <{src['url']}>")
                if len(research_result.sources_used) > 3:
                    brain_content_parts.append(f"   (...and {len(research_result.sources_used) - 3} more sources)")
                brain_content_parts.append("---")

            if research_result.follow_up_questions:
                brain_content_parts.append("## Suggested Follow-up Questions:")
                for q_item in research_result.follow_up_questions:
                    brain_content_parts.append(f"- {q_item}")
                brain_content_parts.append("---")

            if research_result.related_topics:
                brain_content_parts.append(f"## Related Topics/Keywords: {', '.join(research_result.related_topics)}")
                brain_content_parts.append("---")
            
            if research_result.context_used:
                brain_content_parts.append(f"## Original Query Context:\n{research_result.context_used}")

            # Metadata for the brain context
            brain_meta = {
                "original_query": research_result.query,
                "timestamp_utc": research_result.timestamp.isoformat(),
                "user_id": research_result.user_id,
                "confidence_score": research_result.confidence_score,
                "research_duration_sec": research_result.research_time,
                "num_sources_synthesized": len(research_result.sources_used),
                "num_follow_ups": len(research_result.follow_up_questions),
                "primary_keywords": research_result.related_topics[:5], # Top 5 keywords
                "has_user_context": bool(research_result.context_used),
                "tags": ["research", brain_context_type] + [topic.lower().replace(" ","_") for topic in research_result.related_topics[:3]]
            }
            
            # Create a unique, descriptive context name
            query_slug = re.sub(r'\W+', '_', research_result.query.lower())[:50] # Sanitize query for name
            context_name = f"{brain_context_type}_{query_slug}_{research_result.timestamp.strftime('%Y%m%d%H%M')}"

            return {
                "context_name": context_name,
                "operation_type": "upsert_context", # Common operation for brain systems
                "content_format": "markdown",
                "full_text_content": "\n\n".join(brain_content_parts),
                "metadata_dict": brain_meta,
                # "associated_chunks_or_embeddings": [] # If brain links to vector store items
            }
        except Exception as e_brain:
            print(f"❌ Error formatting research for brain system: {e_brain}")
            return {}


    async def save_to_vector_store(self,
                                   research_result_obj: ResearchResult, # --- Parameter name updated ---
                                   vector_store_instance: MLXVectorStore, # --- Parameter name updated ---
                                   model_id: Optional[str] = None) -> bool: # --- model_id can be optional, use config ---
        """Speichert Research Results (WebContent items) in Vector Store."""
        # --- This method was mostly fine, ensuring embeddings exist and using configured model_id ---
        if not vector_store_instance:
            print("⚠️ Vector store instance not provided, cannot save.")
            return False
        if not self.embedding_engine:
            print("⚠️ Embedding engine not available, cannot prepare vectors for saving.")
            return False
        
        effective_model_id = model_id or self.config.embedding_model
        
        vectors_to_add: List[List[float]] = []
        metadata_to_add: List[Dict[str, Any]] = []

        # 1. Embed and store the synthesized answer itself
        if research_result_obj.synthesized_answer:
            answer_text_for_embedding = f"Question: {research_result_obj.query}\nAnswer: {research_result_obj.synthesized_answer}"
            answer_embedding_result = await self.embedding_engine.embed([answer_text_for_embedding])
            if answer_embedding_result and answer_embedding_result.embeddings:
                ans_emb = answer_embedding_result.embeddings[0]
                vectors_to_add.append(ans_emb.tolist() if hasattr(ans_emb, 'tolist') else ans_emb)
                metadata_to_add.append({
                    "type": "synthesized_answer", "original_query": research_result_obj.query,
                    "text_content": research_result_obj.synthesized_answer, "confidence": research_result_obj.confidence_score,
                    "timestamp_utc": research_result_obj.timestamp.isoformat(), "user_id": research_result_obj.user_id,
                    "source_urls": [s['url'] for s in research_result_obj.sources_used],
                    "reference_id": f"answer_{hashlib.md5(research_result_obj.query.encode()).hexdigest()}"
                })
        
        # 2. Embed and store individual WebContent items (if they have embeddings)
        for wc_item in research_result_obj.search_results: # These are the re-ranked, relevant items
            if wc_item.embedding: # Must have an embedding
                vectors_to_add.append(wc_item.embedding)
                
                # Text content for metadata should be what the embedding represents
                meta_text = f"Title: {wc_item.title}\nURL: {wc_item.url}\n"
                meta_text += f"Summary: {wc_item.summary}\n" if wc_item.summary else f"Content Snippet: {wc_item.content[:500]}...\n"

                metadata_to_add.append({
                    "type": "web_content_source", "source_url": wc_item.url, "source_title": wc_item.title,
                    "text_content": meta_text, # The text corresponding to the embedding
                    "domain": wc_item.domain, "language": wc_item.language,
                    "publish_date_utc": wc_item.publish_date.isoformat() if wc_item.publish_date else None,
                    "word_count": wc_item.word_count, "extraction_method": wc_item.metadata.get("extraction_method"),
                    "related_research_query": research_result_obj.query, "user_id": research_result_obj.user_id,
                    "content_hash": wc_item.content_hash,
                    "reference_id": f"source_{wc_item.content_hash}"
                })
        
        if not vectors_to_add:
            print(f"⚠️ No embeddings found or generated for research query '{research_result_obj.query}'. Nothing to save to vector store.")
            return False

        # Namespace for user and research project/date could be useful
        # e.g., f"user_{research_result_obj.user_id}_research_{research_result_obj.timestamp.strftime('%Y-%m')}"
        vs_namespace = f"research_findings_user_{research_result_obj.user_id}"
        
        try:
            print(f"Saving {len(vectors_to_add)} vectors to vector store namespace '{vs_namespace}'...")
            success = await vector_store_instance.add_vectors(
                user_id=research_result_obj.user_id,
                model_id=effective_model_id,
                vectors=vectors_to_add,
                metadata=metadata_to_add,
                namespace=vs_namespace
            )
            if success:
                print(f"✅ Successfully saved {len(vectors_to_add)} items to vector store.")
            else:
                print(f"❌ Vector store save operation reported failure.")
            return success
        except Exception as e_vs_save:
            print(f"❌ Error during vector store save operation: {e_vs_save}")
            return False


    def get_performance_stats(self) -> Dict[str, Any]:
        """Liefert Performance-Statistiken. Enhanced version."""
        # --- This method was mostly fine, slight enhancements to metrics ---
        avg_research_time_sec = self.total_research_time / self.total_queries if self.total_queries > 0 else 0.0
        total_extractions_attempted = self.successful_extractions + self.failed_extractions
        extraction_success_pct = (self.successful_extractions / total_extractions_attempted * 100) \
                                 if total_extractions_attempted > 0 else 0.0
        pages_per_query_avg = self.total_pages_extracted / self.total_queries if self.total_queries > 0 else 0.0

        return {
            "total_research_queries_processed": self.total_queries,
            "total_unique_pages_content_extracted": self.total_pages_extracted, # Sum of len(web_contents) from successful researches
            "total_successful_page_extractions": self.successful_extractions, # Individual extraction attempts
            "total_failed_page_extractions": self.failed_extractions,
            "page_extraction_success_rate_percent": round(extraction_success_pct, 2),
            "average_pages_extracted_per_query": round(pages_per_query_avg, 2),
            "total_cumulative_research_time_seconds": round(self.total_research_time, 3),
            "average_time_per_research_query_seconds": round(avg_research_time_sec, 3),
            "content_cache_current_size": len(self.content_cache),
            "configured_search_engines": list(self.search_engines.keys()),
            "configured_extraction_methods_order": list(self.extractors.keys()),
            "javascript_rendering_enabled": self.config.enable_javascript,
            "auto_summarization_enabled": self.config.auto_summarize,
        }


    async def export_research_result(self,
                                     research_result_obj: ResearchResult, # --- Parameter name updated ---
                                     output_path: Union[str, Path],
                                     output_format: str = "json") -> bool: # --- Parameter name 'format' to 'output_format' ---
        """Exportiert Research Result in JSON oder Markdown."""
        # --- This method was mostly fine, ensuring output path exists ---
        output_path = Path(output_path)
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if output_format.lower() == "json":
                # Use asdict for nested dataclasses, then custom encoder for datetime/Path
                class CustomJsonEncoder(json.JSONEncoder):
                    def default(self, o):
                        if isinstance(o, datetime):
                            return o.isoformat()
                        if isinstance(o, Path): # Though not expected in ResearchResult directly
                            return str(o)
                        if isinstance(o, set): # If any sets are used
                            return list(o)
                        return super().default(o)
                
                # Convert ResearchResult (which contains List[WebContent]) to dicts
                # WebContent itself is a dataclass, so asdict will handle it.
                export_data_dict = asdict(research_result_obj)

                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(export_data_dict, indent=2, ensure_ascii=False, cls=CustomJsonEncoder))

            elif output_format.lower() == "markdown":
                md_parts = [
                    f"# Research Report: {research_result_obj.query}",
                    f"**Date:** {research_result_obj.timestamp.strftime('%Y-%m-%d %H:%M')} | **User:** {research_result_obj.user_id}",
                    f"**Confidence:** {research_result_obj.confidence_score:.2f} | **Time Taken:** {research_result_obj.research_time:.1f}s",
                    "---",
                    "## Synthesized Answer",
                    research_result_obj.synthesized_answer,
                    "---",
                ]
                if research_result_obj.sources_used:
                    md_parts.append("## Key Sources Used in Answer:")
                    for i, src in enumerate(research_result_obj.sources_used):
                        md_parts.append(f"{i+1}. **{src['title']}** ({src['domain']})\n   - URL: <{src['url']}>")
                    md_parts.append("---")
                
                if research_result_obj.follow_up_questions:
                    md_parts.append("## Suggested Follow-up Questions:")
                    for q_item in research_result_obj.follow_up_questions: md_parts.append(f"- {q_item}")
                    md_parts.append("---")

                if research_result_obj.related_topics:
                    md_parts.append(f"## Related Topics: {', '.join(research_result_obj.related_topics)}")
                    md_parts.append("---")

                if research_result_obj.search_results: # These are the top re-ranked WebContent objects
                    md_parts.append("## Top Relevant Web Content Found:")
                    for i, wc_item in enumerate(research_result_obj.search_results[:5]): # Show details for top 5
                        md_parts.append(f"### {i+1}. {wc_item.title}")
                        md_parts.append(f"- **URL:** <{wc_item.url}>")
                        md_parts.append(f"- **Domain:** {wc_item.domain}")
                        if wc_item.publish_date: md_parts.append(f"- **Published:** {wc_item.publish_date.strftime('%Y-%m-%d')}")
                        if wc_item.author: md_parts.append(f"- **Author(s):** {wc_item.author}")
                        md_parts.append(f"- **Word Count:** ~{wc_item.word_count}")
                        md_parts.append(f"- **Extraction Method:** {wc_item.metadata.get('extraction_method', 'N/A')}")
                        if wc_item.summary:
                            md_parts.append(f"- **Summary:**\n  ```\n  {wc_item.summary}\n  ```")
                        elif wc_item.content:
                             md_parts.append(f"- **Content Snippet (first 300 chars):**\n  ```\n  {wc_item.content[:300]}...\n  ```")
                        md_parts.append("") # Spacer
                
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    await f.write("\n\n".join(md_parts))
            else:
                raise ValueError(f"Unsupported export format: {output_format}. Choose 'json' or 'markdown'.")
            
            print(f"✅ Research result exported to {output_path} (Format: {output_format.lower()})")
            return True
        except Exception as e_export:
            print(f"❌ Export research error to {output_path}: {e_export}")
            return False


    async def benchmark(self, test_queries: Optional[List[str]] = None, max_results_per_query: int = 3) -> Dict[str, Any]: # --- Type hint, added max_results ---
        """Performance Benchmark für Research Assistant. Returns dict with aggregated stats."""
        # --- This method was mostly fine, more detailed results collection ---
        print("🚀 Running Research Assistant Benchmark...")
        if not test_queries:
            test_queries = [ # More diverse queries
                "What is the future of artificial general intelligence?",
                "Explain the main components of a blockchain.",
                "Impact of climate change on global food security.",
                "Recent breakthroughs in cancer research.",
                "Summarize the plot of the book 'Dune' by Frank Herbert."
            ]
        
        all_query_metrics: List[Dict[str, Any]] = []
        total_benchmark_start_time = time.perf_counter()

        for i, query_text in enumerate(test_queries):
            print(f"\nBenchmarking query {i+1}/{len(test_queries)}: '{query_text}'")
            query_obj = SearchQuery(
                query=query_text,
                user_id="benchmark_system_user",
                max_results=max_results_per_query, # Use parameter for benchmark search depth
                search_engines=["duckduckgo"] # Use a single, faster engine for benchmark consistency usually
            )
            
            q_start_time = time.perf_counter()
            try:
                research_res = await self.research(query_obj)
                q_duration = time.perf_counter() - q_start_time
                
                all_query_metrics.append({
                    "query": query_text, "duration_sec": q_duration,
                    "num_sources_synthesized": len(research_res.sources_used),
                    "num_pages_extracted": len(research_res.search_results), # web_contents that were extracted & reranked
                    "answer_length_chars": len(research_res.synthesized_answer),
                    "confidence": research_res.confidence_score,
                    "num_follow_ups": len(research_res.follow_up_questions)
                })
                print(f"Query '{query_text}' completed in {q_duration:.2f}s. Confidence: {research_res.confidence_score:.2f}")
            except Exception as e_bench_q:
                q_duration = time.perf_counter() - q_start_time
                print(f"❌ Error benchmarking query '{query_text}' (took {q_duration:.2f}s): {e_bench_q}")
                all_query_metrics.append({"query": query_text, "duration_sec": q_duration, "error": str(e_bench_q)})
        
        total_benchmark_duration = time.perf_counter() - total_benchmark_start_time
        
        if not all_query_metrics:
            return {"error": "No benchmark queries were successfully processed.", "total_duration_sec": total_benchmark_duration}

        # Aggregate results
        successful_queries = [m for m in all_query_metrics if "error" not in m]
        num_successful = len(successful_queries)

        if num_successful == 0:
             return {"error": "All benchmark queries failed.", "total_duration_sec": total_benchmark_duration, "details": all_query_metrics}

        avg_duration = sum(m['duration_sec'] for m in successful_queries) / num_successful
        avg_sources = sum(m['num_sources_synthesized'] for m in successful_queries) / num_successful
        avg_pages = sum(m['num_pages_extracted'] for m in successful_queries) / num_successful
        avg_answer_len = sum(m['answer_length_chars'] for m in successful_queries) / num_successful
        avg_confidence = sum(m['confidence'] for m in successful_queries) / num_successful
        
        final_benchmark_summary = {
            "total_queries_attempted_in_benchmark": len(test_queries),
            "successful_queries_in_benchmark": num_successful,
            "total_benchmark_wall_time_seconds": round(total_benchmark_duration, 3),
            "average_query_wall_time_seconds": round(avg_duration, 3),
            "average_sources_used_per_answer": round(avg_sources, 2),
            "average_pages_extracted_per_query": round(avg_pages, 2),
            "average_answer_length_chars": round(avg_answer_len, 0),
            "average_answer_confidence": round(avg_confidence, 3),
            "queries_per_minute_rate": round(num_successful / (total_benchmark_duration / 60.0), 2) if total_benchmark_duration > 0 else 0,
            "individual_query_metrics (first 3)": all_query_metrics[:3] # Sample of detailed metrics
        }
        print("\n--- Benchmark Summary ---")
        print(json.dumps(final_benchmark_summary, indent=2))
        return final_benchmark_summary


# Example Usage (Original code was here)
# Minor updates to ensure it runs with changes, and to show explicit init/close.
async def example_usage():
    """Beispiele für Research Assistant Usage"""
    print("🚀 Starting MLX Research Assistant Example Usage...")

    # --- Mock MLX Components if not fully set up ---
    class MockEmbeddingEngine:
        async def initialize(self): print("MockEmbeddingEngine initialized.")
        async def embed(self, texts: List[str]):
            class EmbRes: public: embeddings = [[0.1] * 10 for _ in texts] # Dummy
            return EmbRes() if texts else type('obj', (object,), {'embeddings': []})()

    class MockLLMHandler:
        async def initialize(self): print("MockLLMHandler initialized.")
        async def generate_single(self, request: LLMRequest):
            class LLMRes: public: response = f"Mock LLM answer to: {request.prompt[:50]}..."
            return LLMRes()

    class MockRerankEngine:
        async def initialize(self): print("MockRerankEngine initialized.")
        async def rerank(self, query, candidates, top_k, algorithm=None): # algorithm arg might be there
            class RerankRes: public: candidates = candidates[:top_k] # Just take top_k as mock
            return RerankRes()
    
    class MockDocProcessor:
         def __init__(self, cfg, emb_engine): print("MockDocProcessor initialized.")
         async def process_text_directly(self, text, title, user_id, metadata):
             class ProcDoc: public: summary=f"Mock summary for {title}"; chunks=[type('obj', (object,), {'embedding': None})()]
             return ProcDoc()


    research_config = ResearchConfig(
        max_search_results=10, # Fewer for faster example
        auto_summarize=True,
        enable_javascript=False, # Keep False for quick local tests unless Selenium is fully configured
        # embedding_model="mlx-community/gte-small", # Define even if using mock
        # llm_model="mlx-community/gemma-2-9b-it-4bit",  # Define even if using mock
        rate_limit_delay=0.2 # Faster for example
    )

    # Provide mock components to the assistant for this example
    assistant = MLXResearchAssistant(
        config=research_config,
        embedding_engine=MockEmbeddingEngine(), # type: ignore
        llm_handler=MockLLMHandler(),           # type: ignore
        rerank_engine=MockRerankEngine(),         # type: ignore
        document_processor=MockDocProcessor(None, None) # type: ignore
    )

    # It's good practice to explicitly initialize and close.
    await assistant.initialize()

    try:
        print("\n--- Performing Full Research Example ---")
        main_search_query = SearchQuery(
            query="What are the latest advancements in renewable energy storage?",
            user_id="example_user_001",
            max_results=5, # Max results per search engine
            search_engines=["duckduckgo"] # Use one for faster example
        )
        research_report = await assistant.research(main_search_query)

        print(f"\nResearch Report for: '{research_report.query}'")
        print(f"  Synthesized Answer (snippet): {research_report.synthesized_answer[:250]}...")
        print(f"  Sources Cited: {len(research_report.sources_used)}")
        print(f"  Confidence: {research_report.confidence_score:.3f}")
        print(f"  Time Taken: {research_report.research_time:.2f}s")
        if research_report.follow_up_questions:
            print(f"  Follow-up Question 1: {research_report.follow_up_questions[0]}")

        # Export example
        await assistant.export_research_result(research_report, "example_research_report.json", "json")
        await assistant.export_research_result(research_report, "example_research_report.md", "markdown")
        print("  Research report exported to JSON and Markdown.")

        print("\n--- Performing Quick Search Example ---")
        quick_search_results = await assistant.quick_search(
            query="Key features of Python 3.12",
            user_id="example_user_002",
            max_results=2
        )
        print(f"Quick search returned {len(quick_search_results)} WebContent items:")
        for idx, wc_item in enumerate(quick_search_results):
            print(f"  {idx+1}. {wc_item.title} ({wc_item.domain}) - Summary: {'Yes' if wc_item.summary else 'No'}")

        # --- Brain Integration Example ---
        # brain_context_data = await assistant.save_research_to_brain(research_report, "general_research_topic")
        # print(f"\nFormatted data for Brain System (context: '{brain_context_data.get('context_name')}'):")
        # print(f"  Content (first 100 chars): {brain_context_data.get('content', '')[:100]}...")
        # print(f"  Metadata keys: {list(brain_context_data.get('metadata', {}).keys())}")


        # --- Vector Store Saving Example (Mock) ---
        # class MockVectorStore:
        #     async def add_vectors(self, user_id, model_id, vectors, metadata, namespace):
        #         print(f"MockVectorStore: Add {len(vectors)} vectors to '{namespace}'.")
        #         return True
        # mock_vs = MockVectorStore()
        # await assistant.save_to_vector_store(research_report, mock_vs) # type: ignore

    except Exception as e_example:
        print(f"💥 An error occurred in example_usage: {e_example}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n--- Closing Research Assistant Resources ---")
        await assistant.close() # Important to close sessions and drivers

    # --- Performance Stats & Benchmark ---
    # These are typically called after some usage or as separate diagnostic runs.
    # For the example, let's call them here.
    # current_stats = assistant.get_performance_stats()
    # print("\n--- Final Performance Stats from Assistant Instance ---")
    # print(json.dumps(current_stats, indent=2))

    # print("\n--- Running Benchmark ---")
    # benchmark_summary = await assistant.benchmark(test_queries=["What is MLX?"], max_results_per_query=2)
    # print("Benchmark finished.")


if __name__ == "__main__":
    # Note: For Selenium to work, chromedriver must be in PATH or its path specified.
    # You can set assistant.chrome_driver_path = "/path/to/chromedriver" if needed.
    asyncio.run(example_usage())