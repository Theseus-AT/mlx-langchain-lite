#!/usr/bin/env python3
"""
Copyright (c) 2024 Theseus AI Technologies. All rights reserved.

This software is proprietary and confidential. Commercial use requires 
a valid enterprise license from Theseus AI Technologies.

Enterprise Contact: enterprise@theseus-ai.com
Website: https://theseus-ai.com/enterprise
"""

"""
MLX-LangChain-Lite - Zentrales Configuration Management System
Einheitliche Konfiguration für alle Module mit Environment-Support und Validation
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceMode(Enum):
    """Performance-Modi für verschiedene Use Cases"""
    DEVELOPMENT = "development"
    FAST = "fast"
    BALANCED = "balanced" 
    THROUGHPUT = "throughput"
    PRODUCTION = "production"


class LogLevel(Enum):
    """Logging-Level"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ModelConfig:
    """Model-spezifische Konfiguration"""
    # LLM Settings
    llm_model: str = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    llm_dtype: str = "float16"
    llm_trust_remote_code: bool = True
    
    # Embedding Settings
    embedding_model: str = "mlx-community/gte-small"
    embedding_dim: int = 384
    embedding_normalize: bool = True
    embedding_pooling: str = "mean"
    
    # Generation Settings
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: Optional[int] = None
    
    # Model Loading
    model_cache_dir: Optional[str] = None
    auto_download: bool = True


@dataclass
class BatchProcessingConfig:
    """Batch-Processing Konfiguration"""
    # Batch Sizes
    max_batch_size: int = 16
    llm_batch_size: int = 8
    embedding_batch_size: int = 128
    
    # Performance
    auto_batch_size: bool = True
    memory_threshold: float = 0.8
    batch_timeout: float = 30.0
    
    # MLX Parallels
    enable_batch_processing: bool = True
    performance_mode: PerformanceMode = PerformanceMode.BALANCED
    
    # Concurrency
    max_concurrent_batches: int = 4
    queue_timeout: float = 5.0


@dataclass 
class VectorStoreConfig:
    """Vector Store Konfiguration"""
    # Connection
    base_url: str = "http://localhost:8000"
    api_key: str = "mlx-vector-db-key"
    timeout: float = 30.0
    max_retries: int = 3
    
    # User Management
    default_user_id: str = "default_user"
    default_model_id: str = "gte-small"
    
    # Performance
    max_batch_size: int = 100
    connection_pool_size: int = 10
    
    # Storage
    auto_create_stores: bool = True
    backup_enabled: bool = False


@dataclass
class RerankConfig:
    """Reranking Konfiguration"""
    # Method Selection
    primary_method: str = "hybrid"  # "bm25", "tfidf", "llm_scoring", "hybrid"
    fallback_method: str = "bm25"
    
    # LLM Reranking
    enable_llm_reranking: bool = True
    max_docs_for_llm: int = 20
    llm_temperature: float = 0.1
    
    # Scoring
    score_threshold: float = 0.1
    diversity_factor: float = 0.1
    
    # Performance
    enable_caching: bool = True
    cache_size: int = 1000
    
    # BM25/TF-IDF
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class DocumentProcessingConfig:
    """Document Processing Konfiguration"""
    # File Processing
    max_file_size_mb: int = 100
    supported_formats: List[str] = field(default_factory=lambda: [
        "pdf", "docx", "txt", "md", "html", "csv"
    ])
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 100
    adaptive_chunking: bool = True
    
    # OCR Settings
    enable_ocr: bool = False
    ocr_language: str = "eng"
    
    # Metadata Extraction
    extract_metadata: bool = True
    include_page_numbers: bool = True
    include_timestamps: bool = True
    
    # PII Filtering
    enable_pii_filtering: bool = False
    pii_confidence_threshold: float = 0.8


@dataclass
class CodeAnalysisConfig:
    """Code Analysis Konfiguration"""
    # Supported Languages
    supported_languages: List[str] = field(default_factory=lambda: [
        "python", "javascript", "typescript", "java", "cpp", "rust"
    ])
    
    # Tree-sitter
    tree_sitter_grammar_dir: str = "grammars"
    auto_download_grammars: bool = True
    
    # Analysis Settings
    max_file_size_kb: int = 1024
    include_comments: bool = True
    complexity_analysis: bool = True
    
    # Repository Analysis
    max_files_per_repo: int = 1000
    ignore_patterns: List[str] = field(default_factory=lambda: [
        "*.pyc", "node_modules", ".git", "__pycache__"
    ])


@dataclass
class ResearchConfig:
    """Research Assistant Konfiguration"""
    # Web Search
    enable_web_search: bool = True
    max_search_results: int = 10
    search_timeout: float = 30.0
    
    # Selenium Settings
    enable_javascript: bool = False
    webdriver_path: Optional[str] = None
    headless_browser: bool = True
    
    # Content Extraction
    max_content_length: int = 10000
    extract_main_content: bool = True
    remove_ads: bool = True
    
    # Rate Limiting
    request_delay: float = 1.0
    max_concurrent_requests: int = 5
    
    # User Agent
    user_agent: str = "MLX-LangChain-Lite Research Assistant"


@dataclass
class CachingConfig:
    """Caching System Konfiguration"""
    # Global Caching
    enable_global_cache: bool = True
    cache_backend: str = "memory"  # "memory", "redis", "file"
    
    # Memory Cache
    max_memory_cache_size: int = 1000
    cache_ttl_seconds: int = 3600
    
    # File Cache
    file_cache_dir: str = ".cache"
    max_file_cache_size_mb: int = 500
    
    # Redis Cache (falls verwendet)
    redis_url: Optional[str] = None
    redis_db: int = 0
    
    # Cache Policies
    auto_cleanup: bool = True
    cleanup_interval: int = 300


@dataclass
class LoggingConfig:
    """Logging System Konfiguration"""
    # Basic Settings
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File Logging
    log_to_file: bool = True
    log_file: str = "logs/mlx_langchain_lite.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    
    # Console Logging
    log_to_console: bool = True
    console_level: LogLevel = LogLevel.INFO
    
    # Structured Logging
    structured_logs: bool = True
    include_request_id: bool = True
    
    # Performance Logging
    log_performance: bool = True
    log_slow_queries: bool = True
    slow_query_threshold: float = 1.0


@dataclass
class SecurityConfig:
    """Security & Privacy Konfiguration"""
    # API Security
    api_keys: Dict[str, str] = field(default_factory=dict)
    enable_api_key_rotation: bool = False
    
    # Data Privacy
    anonymize_logs: bool = True
    data_retention_days: int = 30
    
    # PII Protection
    enable_pii_detection: bool = True
    pii_masking: bool = True
    
    # Rate Limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 100
    
    # Validation
    input_sanitization: bool = True
    max_input_length: int = 10000


@dataclass
class MonitoringConfig:
    """Performance Monitoring Konfiguration"""
    # Metrics Collection
    enable_metrics: bool = True
    metrics_backend: str = "prometheus"  # "prometheus", "statsd", "internal"
    
    # Performance Tracking
    track_response_times: bool = True
    track_memory_usage: bool = True
    track_token_counts: bool = True
    
    # Health Checks
    enable_health_checks: bool = True
    health_check_interval: int = 60
    
    # Alerting
    enable_alerting: bool = False
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "response_time_ms": 5000,
        "memory_usage_percent": 90,
        "error_rate_percent": 10
    })
    
    # Dashboards
    enable_web_dashboard: bool = False
    dashboard_port: int = 8080


@dataclass
class SystemConfig:
    """Haupt-System-Konfiguration"""
    # Meta Information
    version: str = "1.0.0"
    environment: str = "development"  # "development", "staging", "production"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Component Configurations
    models: ModelConfig = field(default_factory=ModelConfig)
    batch_processing: BatchProcessingConfig = field(default_factory=BatchProcessingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    reranking: RerankConfig = field(default_factory=RerankConfig)
    document_processing: DocumentProcessingConfig = field(default_factory=DocumentProcessingConfig)
    code_analysis: CodeAnalysisConfig = field(default_factory=CodeAnalysisConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Global Settings
    debug: bool = False
    verbose: bool = False
    dry_run: bool = False
    
    # Paths
    data_dir: str = "data"
    cache_dir: str = ".cache"
    logs_dir: str = "logs"
    models_dir: str = "models"


class ConfigManager:
    """
    Zentraler Configuration Manager
    
    Features:
    - Environment Variable Support
    - YAML/JSON Configuration Files
    - Runtime Configuration Updates
    - Validation and Defaults
    - Configuration Profiles
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or "config")
        self.config_dir.mkdir(exist_ok=True)
        
        self.config: Optional[SystemConfig] = None
        self.config_file: Optional[Path] = None
        
        # Environment Variable Mapping
        self.env_mapping = self._create_env_mapping()
        
        logger.info(f"ConfigManager initialisiert - Config Dir: {self.config_dir}")
    
    def load_config(
        self,
        config_file: Optional[str] = None,
        profile: Optional[str] = None
    ) -> SystemConfig:
        """Lädt Konfiguration aus Datei und Environment Variables"""
        
        try:
            # 1. Basis-Konfiguration erstellen
            config = SystemConfig()
            
            # 2. Configuration File laden falls vorhanden
            if config_file:
                config_path = self.config_dir / config_file
                if config_path.exists():
                    file_config = self._load_config_file(config_path)
                    config = self._merge_configs(config, file_config)
                    self.config_file = config_path
                    logger.info(f"Konfiguration geladen aus: {config_path}")
            
            # 3. Profile-spezifische Konfiguration
            if profile:
                profile_config = self._load_profile_config(profile)
                if profile_config:
                    config = self._merge_configs(config, profile_config)
                    logger.info(f"Profile '{profile}' angewendet")
            
            # 4. Environment Variables anwenden
            config = self._apply_environment_variables(config)
            
            # 5. Auto-Konfiguration basierend auf Environment
            config = self._apply_auto_configuration(config)
            
            # 6. Validierung
            self._validate_config(config)
            
            self.config = config
            logger.info("✅ Konfiguration erfolgreich geladen")
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Konfiguration laden fehlgeschlagen: {e}")
            raise
    
    def save_config(
        self,
        config: Optional[SystemConfig] = None,
        filename: str = "system_config.yaml"
    ) -> bool:
        """Speichert aktuelle Konfiguration"""
        
        config = config or self.config
        if not config:
            raise ValueError("Keine Konfiguration zum Speichern verfügbar")
        
        try:
            config_path = self.config_dir / filename
            
            # Konvertiere zu Dict und bereinige
            config_dict = asdict(config)
            config_dict = self._clean_config_for_export(config_dict)
            
            # Als YAML speichern
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Konfiguration gespeichert: {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Konfiguration speichern fehlgeschlagen: {e}")
            return False
    
    def get_config(self) -> SystemConfig:
        """Gibt aktuelle Konfiguration zurück"""
        if not self.config:
            return self.load_config()
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Aktualisiert Konfiguration zur Laufzeit"""
        
        if not self.config:
            raise ValueError("Keine Konfiguration geladen")
        
        try:
            # Flache Updates anwenden
            for key, value in updates.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                else:
                    # Verschachtelte Updates
                    self._apply_nested_update(self.config, key, value)
            
            # Re-validierung
            self._validate_config(self.config)
            
            logger.info(f"✅ Konfiguration aktualisiert: {list(updates.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Konfiguration Update fehlgeschlagen: {e}")
            return False
    
    def create_profile(
        self,
        profile_name: str,
        base_config: Optional[SystemConfig] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Erstellt neues Konfiguration-Profil"""
        
        try:
            base_config = base_config or self.get_config()
            config_dict = asdict(base_config)
            
            # Overrides anwenden
            if overrides:
                config_dict.update(overrides)
            
            # Profile speichern
            profile_path = self.config_dir / f"profile_{profile_name}.yaml"
            with open(profile_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Profil '{profile_name}' erstellt: {profile_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Profil erstellen fehlgeschlagen: {e}")
            return False
    
    def list_profiles(self) -> List[str]:
        """Listet verfügbare Profile auf"""
        profiles = []
        for file_path in self.config_dir.glob("profile_*.yaml"):
            profile_name = file_path.stem.replace("profile_", "")
            profiles.append(profile_name)
        return sorted(profiles)
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Lädt Konfiguration aus Datei"""
        
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                return json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    def _load_profile_config(self, profile: str) -> Optional[Dict[str, Any]]:
        """Lädt Profile-spezifische Konfiguration"""
        
        profile_path = self.config_dir / f"profile_{profile}.yaml"
        if profile_path.exists():
            return self._load_config_file(profile_path)
        return None
    
    def _merge_configs(self, base: SystemConfig, override: Dict[str, Any]) -> SystemConfig:
        """Merged zwei Konfigurationen"""
        
        # Konvertiere base zu Dict
        base_dict = asdict(base)
        
        # Deep merge
        merged_dict = self._deep_merge(base_dict, override)
        
        # Zurück zu SystemConfig
        return self._dict_to_config(merged_dict)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge von zwei Dictionaries"""
        
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> SystemConfig:
        """Konvertiert Dictionary zu SystemConfig"""
        
        # Vereinfachte Konvertierung - könnte durch pydantic ersetzt werden
        try:
            return SystemConfig(**config_dict)
        except Exception as e:
            logger.warning(f"Config conversion error: {e}")
            # Fallback: Basis-Config mit partiellen Updates
            config = SystemConfig()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return config
    
    def _create_env_mapping(self) -> Dict[str, str]:
        """Erstellt Mapping zwischen Environment Variables und Config-Feldern"""
        
        return {
            # Models
            "MLX_LLM_MODEL": "models.llm_model",
            "MLX_EMBEDDING_MODEL": "models.embedding_model",
            "MLX_MAX_TOKENS": "models.max_tokens",
            "MLX_TEMPERATURE": "models.temperature",
            
            # Vector Store
            "VECTOR_DB_URL": "vector_store.base_url",
            "VECTOR_DB_API_KEY": "vector_store.api_key",
            "VECTOR_DB_DEFAULT_USER": "vector_store.default_user_id",
            
            # Batch Processing
            "MLX_BATCH_SIZE": "batch_processing.max_batch_size",
            "MLX_PERFORMANCE_MODE": "batch_processing.performance_mode",
            "MLX_ENABLE_BATCH": "batch_processing.enable_batch_processing",
            
            # Logging
            "LOG_LEVEL": "logging.level",
            "LOG_FILE": "logging.log_file",
            "LOG_TO_CONSOLE": "logging.log_to_console",
            
            # Security
            "API_KEY": "security.api_keys.main",
            "ENABLE_PII_DETECTION": "security.enable_pii_detection",
            
            # System
            "ENVIRONMENT": "environment",
            "DEBUG": "debug",
            "DATA_DIR": "data_dir",
            "CACHE_DIR": "cache_dir"
        }
    
    def _apply_environment_variables(self, config: SystemConfig) -> SystemConfig:
        """Wendet Environment Variables auf Konfiguration an"""
        
        for env_var, config_path in self.env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Nested path navigation
                    parts = config_path.split('.')
                    obj = config
                    
                    # Navigate to parent object
                    for part in parts[:-1]:
                        obj = getattr(obj, part)
                    
                    # Set final value with type conversion
                    final_key = parts[-1]
                    current_value = getattr(obj, final_key)
                    
                    # Type conversion based on current value type
                    if isinstance(current_value, bool):
                        converted_value = env_value.lower() in ['true', '1', 'yes', 'on']
                    elif isinstance(current_value, int):
                        converted_value = int(env_value)
                    elif isinstance(current_value, float):
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value
                    
                    setattr(obj, final_key, converted_value)
                    logger.debug(f"Environment variable applied: {env_var} -> {config_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to apply env var {env_var}: {e}")
        
        return config
    
    def _apply_auto_configuration(self, config: SystemConfig) -> SystemConfig:
        """Automatische Konfiguration basierend auf Environment"""
        
        # Development vs Production optimizations
        if config.environment == "production":
            config.debug = False
            config.verbose = False
            config.logging.level = LogLevel.WARNING
            config.batch_processing.performance_mode = PerformanceMode.PRODUCTION
            config.security.enable_pii_detection = True
            config.monitoring.enable_metrics = True
            
        elif config.environment == "development":
            config.debug = True
            config.verbose = True
            config.logging.level = LogLevel.DEBUG
            config.logging.log_to_console = True
            config.batch_processing.performance_mode = PerformanceMode.DEVELOPMENT
            
        # Apple Silicon optimizations
        try:
            import platform
            if 'arm' in platform.processor().lower() or 'Apple' in platform.processor():
                config.models.llm_dtype = "float16"
                config.batch_processing.auto_batch_size = True
                logger.info("🍎 Apple Silicon optimizations applied")
        except:
            pass
        
        # Memory-based batch size optimization
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if total_memory_gb < 8:
                config.batch_processing.max_batch_size = 4
                config.batch_processing.embedding_batch_size = 32
            elif total_memory_gb < 16:
                config.batch_processing.max_batch_size = 8
                config.batch_processing.embedding_batch_size = 64
            else:
                config.batch_processing.max_batch_size = 16
                config.batch_processing.embedding_batch_size = 128
                
        except:
            pass
        
        return config
    
    def _apply_nested_update(self, config: SystemConfig, key: str, value: Any):
        """Wendet verschachtelte Updates an"""
        
        if '.' in key:
            parts = key.split('.')
            obj = config
            
            for part in parts[:-1]:
                obj = getattr(obj, part)
            
            setattr(obj, parts[-1], value)
        else:
            setattr(config, key, value)
    
    def _validate_config(self, config: SystemConfig):
        """Validiert Konfiguration"""
        
        # Model validation
        if not config.models.llm_model:
            raise ValueError("LLM model must be specified")
        
        if not config.models.embedding_model:
            raise ValueError("Embedding model must be specified")
        
        # Vector Store validation
        if not config.vector_store.base_url:
            raise ValueError("Vector store base URL must be specified")
        
        if not config.vector_store.api_key:
            logger.warning("⚠️  Vector store API key not set")
        
        # Batch processing validation
        if config.batch_processing.max_batch_size < 1:
            raise ValueError("Batch size must be >= 1")
        
        if config.batch_processing.memory_threshold <= 0 or config.batch_processing.memory_threshold > 1:
            raise ValueError("Memory threshold must be between 0 and 1")
        
        # Paths validation
        for path_attr in ['data_dir', 'cache_dir', 'logs_dir']:
            path_value = getattr(config, path_attr)
            if path_value:
                Path(path_value).mkdir(parents=True, exist_ok=True)
    
    def _clean_config_for_export(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Bereinigt Konfiguration für Export (entfernt sensible Daten)"""
        
        # Sensible Felder maskieren
        sensitive_fields = [
            'api_key', 'api_keys', 'password', 'secret', 'token'
        ]
        
        def clean_recursive(obj):
            if isinstance(obj, dict):
                return {
                    k: '***MASKED***' if any(sensitive in k.lower() for sensitive in sensitive_fields)
                    else clean_recursive(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [clean_recursive(item) for item in obj]
            else:
                return obj
        
        return clean_recursive(config_dict)


# Convenience Functions

def get_config_manager(config_dir: Optional[str] = None) -> ConfigManager:
    """Global Config Manager Singleton"""
    if not hasattr(get_config_manager, '_instance'):
        get_config_manager._instance = ConfigManager(config_dir)
    return get_config_manager._instance


def load_system_config(
    config_file: Optional[str] = None,
    profile: Optional[str] = None,
    config_dir: Optional[str] = None
) -> SystemConfig:
    """Lädt System-Konfiguration"""
    manager = get_config_manager(config_dir)
    return manager.load_config(config_file, profile)


def get_system_config() -> SystemConfig:
    """Gibt aktuelle System-Konfiguration zurück"""
    manager = get_config_manager()
    return manager.get_config()


def create_default_configs():
    """Erstellt Standard-Konfigurationsdateien"""
    
    manager = get_config_manager()
    
    # Development Profile
    dev_overrides = {
        'environment': 'development',
        'debug': True,
        'verbose': True,
        'logging': {
            'level': 'DEBUG',
            'log_to_console': True
        },
        'batch_processing': {
            'performance_mode': 'development',
            'max_batch_size': 4
        }
    }
    manager.create_profile('development', overrides=dev_overrides)
    
    # Production Profile
    prod_overrides = {
        'environment': 'production',
        'debug': False,
        'verbose': False,
        'logging': {
            'level': 'WARNING',
            'log_to_file': True
        },
        'batch_processing': {
            'performance_mode': 'production',
            'max_batch_size': 32
        },
        'security': {
            'enable_pii_detection': True,
            'enable_rate_limiting': True
        },
        'monitoring': {
            'enable_metrics': True,
            'enable_health_checks': True
        }
    }
    manager.create_profile('production', overrides=prod_overrides)
    
    # Fast Inference Profile
    fast_overrides = {
        'batch_processing': {
            'performance_mode': 'fast',
            'max_batch_size': 8
        },
        'models': {
            'max_tokens': 256,
            'temperature': 0.1
        },
        'reranking': {
            'primary_method': 'bm25',
            'enable_llm_reranking': False
        }
    }
    manager.create_profile('fast', overrides=fast_overrides)
    
    # High Throughput Profile
    throughput_overrides = {
        'batch_processing': {
            'performance_mode': 'throughput',
            'max_batch_size': 64,
            'embedding_batch_size': 256
        },
        'vector_store': {
            'max_batch_size': 200
        },
        'reranking': {
            'primary_method': 'hybrid',
            'max_docs_for_llm': 10
        }
    }
    manager.create_profile('throughput', overrides=throughput_overrides)
    
    # Basis-Konfiguration speichern
    base_config = SystemConfig()
    manager.save_config(base_config, "default_config.yaml")
    
    logger.info("✅ Standard-Konfigurationsprofile erstellt")


# Environment Detection
def detect_environment() -> Dict[str, Any]:
    """Erkennt System-Environment und gibt Empfehlungen zurück"""
    
    info = {}
    
    # Hardware Detection
    try:
        import platform
        import psutil
        
        info['platform'] = platform.platform()
        info['processor'] = platform.processor()
        info['machine'] = platform.machine()
        info['total_memory_gb'] = psutil.virtual_memory().total / (1024**3)
        info['cpu_count'] = psutil.cpu_count()
        
        # Apple Silicon Detection
        info['apple_silicon'] = (
            'arm' in platform.processor().lower() or 
            'Apple' in platform.processor() or
            'arm64' in platform.machine().lower()
        )
        
    except Exception as e:
        logger.warning(f"Hardware detection failed: {e}")
        info['detection_error'] = str(e)
    
    # MLX Detection
    try:
        import mlx.core as mx
        info['mlx_available'] = True
        info['mlx_version'] = mx.__version__
        
        # Test MLX functionality
        test_array = mx.random.normal((10, 10))
        mx.eval(test_array)
        info['mlx_functional'] = True
        
    except ImportError:
        info['mlx_available'] = False
        info['mlx_functional'] = False
    except Exception as e:
        info['mlx_available'] = True
        info['mlx_functional'] = False
        info['mlx_error'] = str(e)
    
    # MLX-LM Detection
    try:
        import mlx_lm
        info['mlx_lm_available'] = True
    except ImportError:
        info['mlx_lm_available'] = False
    
    # Performance Recommendations
    recommendations = []
    
    if info.get('apple_silicon'):
        recommendations.append("Use float16 models for better performance")
        recommendations.append("Enable auto batch sizing")
        recommendations.append("Use performance_mode='balanced' or 'fast'")
    
    if info.get('total_memory_gb', 0) < 8:
        recommendations.append("Use smaller batch sizes (4-8)")
        recommendations.append("Consider quantized models")
        recommendations.append("Enable aggressive caching")
    elif info.get('total_memory_gb', 0) > 32:
        recommendations.append("Use larger batch sizes (32-64)")
        recommendations.append("Enable high throughput mode")
    
    if not info.get('mlx_functional'):
        recommendations.append("Fix MLX installation for optimal performance")
        recommendations.append("Consider fallback to CPU-based processing")
    
    info['recommendations'] = recommendations
    
    return info


def validate_system_requirements() -> Dict[str, bool]:
    """Validiert System-Anforderungen"""
    
    requirements = {
        'python_version': True,
        'mlx_available': False,
        'mlx_lm_available': False,
        'sufficient_memory': False,
        'apple_silicon': False
    }
    
    try:
        import sys
        requirements['python_version'] = sys.version_info >= (3, 9)
        
        import mlx.core as mx
        requirements['mlx_available'] = True
        
        # Test MLX
        test = mx.random.normal((5, 5))
        mx.eval(test)
        
        import mlx_lm
        requirements['mlx_lm_available'] = True
        
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        requirements['sufficient_memory'] = memory_gb >= 4
        
        import platform
        proc = platform.processor().lower()
        machine = platform.machine().lower()
        requirements['apple_silicon'] = (
            'arm' in proc or 'apple' in proc or 'arm64' in machine
        )
        
    except Exception as e:
        logger.debug(f"Requirements check error: {e}")
    
    return requirements


# Configuration Validation
def validate_configuration(config: SystemConfig) -> List[str]:
    """Validiert Konfiguration und gibt Warnings/Errors zurück"""
    
    issues = []
    
    # Model validation
    if not config.models.llm_model:
        issues.append("ERROR: LLM model not specified")
    elif not config.models.llm_model.startswith('mlx-community/'):
        issues.append("WARNING: Non-MLX model specified, may have compatibility issues")
    
    # Memory validation
    try:
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        if config.batch_processing.max_batch_size > 32 and available_memory < 16:
            issues.append("WARNING: Large batch size with limited memory may cause OOM")
        
        if config.batch_processing.embedding_batch_size > 256 and available_memory < 8:
            issues.append("WARNING: Large embedding batch size may cause performance issues")
            
    except:
        issues.append("WARNING: Could not detect system memory")
    
    # Vector Store validation
    if 'localhost' in config.vector_store.base_url and config.environment == 'production':
        issues.append("WARNING: Using localhost vector store in production")
    
    if config.vector_store.api_key == 'mlx-vector-db-key':
        issues.append("WARNING: Using default API key, change for security")
    
    # Security validation
    if config.environment == 'production':
        if not config.security.enable_pii_detection:
            issues.append("WARNING: PII detection disabled in production")
        
        if not config.logging.log_to_file:
            issues.append("WARNING: File logging disabled in production")
        
        if config.debug:
            issues.append("WARNING: Debug mode enabled in production")
    
    # Performance validation
    if (config.batch_processing.performance_mode == PerformanceMode.THROUGHPUT and 
        config.batch_processing.max_batch_size < 16):
        issues.append("WARNING: Low batch size for throughput mode")
    
    return issues


# Configuration Templates
def create_development_config() -> SystemConfig:
    """Erstellt Development-optimierte Konfiguration"""
    
    config = SystemConfig()
    config.environment = "development"
    config.debug = True
    config.verbose = True
    
    # Development-optimierte Settings
    config.batch_processing.performance_mode = PerformanceMode.DEVELOPMENT
    config.batch_processing.max_batch_size = 4
    config.logging.level = LogLevel.DEBUG
    config.logging.log_to_console = True
    config.security.enable_pii_detection = False
    config.monitoring.enable_metrics = False
    
    return config


def create_production_config() -> SystemConfig:
    """Erstellt Production-optimierte Konfiguration"""
    
    config = SystemConfig()
    config.environment = "production"
    config.debug = False
    config.verbose = False
    
    # Production-optimierte Settings
    config.batch_processing.performance_mode = PerformanceMode.PRODUCTION
    config.batch_processing.max_batch_size = 32
    config.logging.level = LogLevel.WARNING
    config.logging.log_to_file = True
    config.security.enable_pii_detection = True
    config.security.enable_rate_limiting = True
    config.monitoring.enable_metrics = True
    config.monitoring.enable_health_checks = True
    
    return config


def create_apple_silicon_config() -> SystemConfig:
    """Erstellt Apple Silicon-optimierte Konfiguration"""
    
    config = SystemConfig()
    
    # Apple Silicon Optimierungen
    config.models.llm_dtype = "float16"
    config.batch_processing.auto_batch_size = True
    config.batch_processing.performance_mode = PerformanceMode.BALANCED
    
    # Memory-effiziente Settings
    config.caching.enable_global_cache = True
    config.caching.max_memory_cache_size = 2000
    
    return config


# Configuration Migration
def migrate_old_config(old_config_path: str) -> SystemConfig:
    """Migriert alte Konfiguration zu neuem Format"""
    
    try:
        with open(old_config_path, 'r') as f:
            if old_config_path.endswith('.json'):
                old_config = json.load(f)
            else:
                old_config = yaml.safe_load(f)
        
        # Basis-Konfiguration
        new_config = SystemConfig()
        
        # Migration Mapping
        migration_map = {
            'model_name': 'models.llm_model',
            'embedding_model': 'models.embedding_model',
            'max_tokens': 'models.max_tokens',
            'temperature': 'models.temperature',
            'batch_size': 'batch_processing.max_batch_size',
            'vector_db_url': 'vector_store.base_url',
            'vector_db_key': 'vector_store.api_key',
            'log_level': 'logging.level',
            'debug': 'debug'
        }
        
        # Migriere bekannte Felder
        for old_key, new_path in migration_map.items():
            if old_key in old_config:
                value = old_config[old_key]
                
                # Navigate zu Ziel
                parts = new_path.split('.')
                obj = new_config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                
                setattr(obj, parts[-1], value)
        
        logger.info(f"✅ Konfiguration migriert von: {old_config_path}")
        return new_config
        
    except Exception as e:
        logger.error(f"❌ Migration fehlgeschlagen: {e}")
        raise


# Export Functions
def export_config_schema() -> Dict[str, Any]:
    """Exportiert Konfiguration-Schema für Dokumentation"""
    
    def get_field_info(field_type):
        """Extrahiert Feld-Information"""
        if hasattr(field_type, '__origin__'):
            if field_type.__origin__ is list:
                return {'type': 'array', 'items': get_field_info(field_type.__args__[0])}
            elif field_type.__origin__ is dict:
                return {'type': 'object'}
            elif field_type.__origin__ is Union:
                # Handle Optional types
                non_none_types = [t for t in field_type.__args__ if t != type(None)]
                if len(non_none_types) == 1:
                    return get_field_info(non_none_types[0])
        
        if field_type == str:
            return {'type': 'string'}
        elif field_type == int:
            return {'type': 'integer'}
        elif field_type == float:
            return {'type': 'number'}
        elif field_type == bool:
            return {'type': 'boolean'}
        elif hasattr(field_type, '__dataclass_fields__'):
            return {'type': 'object', 'properties': get_dataclass_schema(field_type)}
        elif isinstance(field_type, type) and issubclass(field_type, Enum):
            return {'type': 'string', 'enum': [e.value for e in field_type]}
        else:
            return {'type': 'string'}
    
    def get_dataclass_schema(dataclass_type):
        """Extrahiert Schema für Dataclass"""
        properties = {}
        for field_name, field in dataclass_type.__dataclass_fields__.items():
            field_info = get_field_info(field.type)
            if field.default != dataclass_type.__dataclass_fields__[field_name].default_factory:
                field_info['default'] = field.default
            properties[field_name] = field_info
        return properties
    
    return {
        'type': 'object',
        'properties': get_dataclass_schema(SystemConfig),
        'title': 'MLX-LangChain-Lite System Configuration',
        'version': '1.0.0'
    }


# CLI Integration
def create_config_cli():
    """Erstellt CLI für Konfiguration-Management"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='MLX-LangChain-Lite Configuration Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create configuration')
    create_parser.add_argument('--profile', choices=['development', 'production', 'fast', 'throughput'])
    create_parser.add_argument('--output', default='config.yaml', help='Output file')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('config_file', help='Configuration file to validate')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate old configuration')
    migrate_parser.add_argument('old_config', help='Old configuration file')
    migrate_parser.add_argument('--output', default='migrated_config.yaml', help='Output file')
    
    # Schema command
    schema_parser = subparsers.add_parser('schema', help='Export configuration schema')
    schema_parser.add_argument('--output', default='config_schema.json', help='Output file')
    
    # Environment command
    env_parser = subparsers.add_parser('env', help='Show environment information')
    
    return parser


def main():
    """CLI Hauptfunktion"""
    
    parser = create_config_cli()
    args = parser.parse_args()
    
    if args.command == 'create':
        if args.profile:
            if args.profile == 'development':
                config = create_development_config()
            elif args.profile == 'production':
                config = create_production_config()
            elif args.profile == 'fast':
                config = SystemConfig()
                config.batch_processing.performance_mode = PerformanceMode.FAST
            elif args.profile == 'throughput':
                config = SystemConfig()
                config.batch_processing.performance_mode = PerformanceMode.THROUGHPUT
        else:
            config = SystemConfig()
        
        manager = ConfigManager()
        manager.save_config(config, args.output)
        print(f"✅ Configuration created: {args.output}")
    
    elif args.command == 'validate':
        manager = ConfigManager()
        config = manager.load_config(args.config_file)
        issues = validate_configuration(config)
        
        if issues:
            print("⚠️  Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Configuration is valid")
    
    elif args.command == 'migrate':
        new_config = migrate_old_config(args.old_config)
        manager = ConfigManager()
        manager.save_config(new_config, args.output)
        print(f"✅ Configuration migrated: {args.output}")
    
    elif args.command == 'schema':
        schema = export_config_schema()
        with open(args.output, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"✅ Schema exported: {args.output}")
    
    elif args.command == 'env':
        env_info = detect_environment()
        print("🔍 Environment Information:")
        for key, value in env_info.items():
            if key != 'recommendations':
                print(f"  {key}: {value}")
        
        if env_info.get('recommendations'):
            print("\n💡 Recommendations:")
            for rec in env_info['recommendations']:
                print(f"  - {rec}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()