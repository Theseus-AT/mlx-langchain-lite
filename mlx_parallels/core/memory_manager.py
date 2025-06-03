"""
MLX Parallels - Memory-Management-Utilities
Optimiert für Apple Silicon Unified Memory und MLX-spezifische Anforderungen
"""

import os
import time
import psutil
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
import threading
import mlx.core as mx
import logging

from .config import MLXParallelsConfig

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot der aktuellen Speichernutzung"""
    timestamp: float
    system_total_gb: float
    system_available_gb: float
    system_used_gb: float
    process_rss_mb: float
    process_vms_mb: float
    mlx_allocated_mb: float
    mlx_peak_mb: float
    gpu_memory_mb: Optional[float] = None


class MemoryManager:
    """
    Memory Manager optimiert für Apple Silicon und MLX
    
    Features:
    - Unified Memory Tracking für Apple Silicon
    - MLX-spezifische Memory-Metriken
    - Automatische Memory-Optimierung
    - Memory Pressure Detection
    - Batch-Size Optimization basierend auf verfügbarem Speicher
    """
    
    def __init__(self, config: MLXParallelsConfig):
        self.config = config
        self.process = psutil.Process()
        
        # Memory tracking
        self.snapshots: List[MemorySnapshot] = []
        self.max_snapshots = 100
        self._lock = threading.Lock()
        
        # Apple Silicon detection
        self.is_apple_silicon = self._detect_apple_silicon()
        self.unified_memory = self.is_apple_silicon
        
        # Memory thresholds
        self.warning_threshold = 0.8  # 80% Memory-Nutzung
        self.critical_threshold = 0.9  # 90% Memory-Nutzung
        
        # MLX Memory tracking
        self.mlx_peak_usage = 0.0
        self.mlx_baseline = 0.0
        
        logger.info(f"MemoryManager initialisiert - Apple Silicon: {self.is_apple_silicon}")
        
        # Baseline Memory-Verwendung messen
        self._establish_baseline()
    
    def _detect_apple_silicon(self) -> bool:
        """Erkennt Apple Silicon Hardware"""
        try:
            import platform
            
            # Platform-Check
            if platform.system() != "Darwin":
                return False
            
            processor = platform.processor().lower()
            machine = platform.machine().lower()
            
            # Apple Silicon Indikatoren
            apple_indicators = ['arm', 'apple', 'aarch64']
            
            for indicator in apple_indicators:
                if indicator in processor or indicator in machine:
                    return True
            
            # Zusätzlicher Check über sysctl (macOS-spezifisch)
            try:
                import subprocess
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if 'Apple' in result.stdout:
                    return True
            except:
                pass
            
            return False
            
        except Exception as e:
            logger.warning(f"Apple Silicon Detection fehlgeschlagen: {e}")
            return False
    
    def _establish_baseline(self):
        """Etabliert Baseline für Memory-Messungen"""
        try:
            # MLX Memory baseline
            if mx.metal.is_available():
                # Kleine MLX Operation um Baseline zu etablieren
                test_array = mx.random.normal((10, 10))
                mx.eval(test_array)
                self.mlx_baseline = self._get_mlx_memory_mb()
            
            # System baseline
            initial_snapshot = self._create_snapshot()
            self.snapshots.append(initial_snapshot)
            
            logger.info(f"Memory Baseline etabliert - MLX: {self.mlx_baseline:.1f}MB")
            
        except Exception as e:
            logger.warning(f"Baseline-Etablierung fehlgeschlagen: {e}")
    
    def _create_snapshot(self) -> MemorySnapshot:
        """Erstellt aktuellen Memory-Snapshot"""
        timestamp = time.time()
        
        # System Memory
        system_memory = psutil.virtual_memory()
        system_total_gb = system_memory.total / (1024**3)
        system_available_gb = system_memory.available / (1024**3)
        system_used_gb = (system_memory.total - system_memory.available) / (1024**3)
        
        # Process Memory
        process_info = self.process.memory_info()
        process_rss_mb = process_info.rss / (1024**2)
        process_vms_mb = process_info.vms / (1024**2)
        
        # MLX Memory
        mlx_allocated_mb = self._get_mlx_memory_mb()
        self.mlx_peak_usage = max(self.mlx_peak_usage, mlx_allocated_mb)
        
        # GPU Memory (falls verfügbar)
        gpu_memory_mb = self._get_gpu_memory_mb()
        
        return MemorySnapshot(
            timestamp=timestamp,
            system_total_gb=system_total_gb,
            system_available_gb=system_available_gb,
            system_used_gb=system_used_gb,
            process_rss_mb=process_rss_mb,
            process_vms_mb=process_vms_mb,
            mlx_allocated_mb=mlx_allocated_mb,
            mlx_peak_mb=self.mlx_peak_usage,
            gpu_memory_mb=gpu_memory_mb
        )
    
    def _get_mlx_memory_mb(self) -> float:
        """Ermittelt aktuelle MLX Memory-Verwendung"""
        try:
            if mx.metal.is_available():
                # MLX Metal Memory
                metal_memory = mx.metal.get_active_memory() / (1024**2)  # MB
                return metal_memory
            else:
                # Fallback für CPU-only
                return 0.0
        except Exception as e:
            logger.debug(f"MLX Memory-Abfrage fehlgeschlagen: {e}")
            return 0.0
    
    def _get_gpu_memory_mb(self) -> Optional[float]:
        """Ermittelt GPU Memory-Verwendung (Apple Silicon spezifisch)"""
        try:
            if self.is_apple_silicon and mx.metal.is_available():
                # Für Apple Silicon ist GPU Memory Teil des Unified Memory
                # Hier könnten spezifische Metal Performance Shaders APIs verwendet werden
                return self._get_mlx_memory_mb()
            return None
        except:
            return None
    
    def get_current_usage(self) -> Dict[str, float]:
        """Gibt aktuelle Memory-Verwendung zurück"""
        with self._lock:
            snapshot = self._create_snapshot()
            self.snapshots.append(snapshot)
            
            # Alte Snapshots entfernen
            if len(self.snapshots) > self.max_snapshots:
                self.snapshots = self.snapshots[-self.max_snapshots:]
            
            return {
                'system_total_gb': snapshot.system_total_gb,
                'system_available_gb': snapshot.system_available_gb,
                'system_used_gb': snapshot.system_used_gb,
                'system_usage_percent': (snapshot.system_used_gb / snapshot.system_total_gb) * 100,
                'process_rss_mb': snapshot.process_rss_mb,
                'process_vms_mb': snapshot.process_vms_mb,
                'mlx_allocated_mb': snapshot.mlx_allocated_mb,
                'mlx_peak_mb': snapshot.mlx_peak_mb,
                'gpu_memory_mb': snapshot.gpu_memory_mb,
                'unified_memory': self.unified_memory
            }
    
    def get_available_memory_mb(self) -> Optional[float]:
        """Gibt verfügbaren Speicher in MB zurück"""
        try:
            current_usage = self.get_current_usage()
            available_gb = current_usage['system_available_gb']
            
            # Sicherheitspuffer abziehen
            safety_buffer_gb = 1.0  # 1GB Puffer
            usable_gb = max(0, available_gb - safety_buffer_gb)
            
            return usable_gb * 1024  # Konvertierung zu MB
            
        except Exception as e:
            logger.warning(f"Available Memory Berechnung fehlgeschlagen: {e}")
            return None
    
    def check_memory_pressure(self) -> Tuple[str, float]:
        """
        Überprüft Memory Pressure
        
        Returns:
            Tuple[status, usage_percent] wo status ist "normal", "warning", "critical"
        """
        usage = self.get_current_usage()
        usage_percent = usage['system_usage_percent'] / 100
        
        if usage_percent >= self.critical_threshold:
            return "critical", usage_percent
        elif usage_percent >= self.warning_threshold:
            return "warning", usage_percent
        else:
            return "normal", usage_percent
    
    def optimize_for_batch_size(self, batch_size: int, estimated_memory_per_item_mb: float) -> int:
        """
        Optimiert Batch-Größe basierend auf verfügbarem Speicher
        
        Args:
            batch_size: Gewünschte Batch-Größe
            estimated_memory_per_item_mb: Geschätzte Memory-Verwendung pro Item
            
        Returns:
            Optimierte Batch-Größe
        """
        available_memory_mb = self.get_available_memory_mb()
        
        if available_memory_mb is None:
            return batch_size
        
        # Memory für aktuellen Prozess und MLX berücksichtigen
        current_usage = self.get_current_usage()
        reserved_memory_mb = (
            current_usage['process_rss_mb'] + 
            current_usage['mlx_allocated_mb'] + 
            500  # Zusätzlicher Puffer
        )
        
        usable_memory_mb = available_memory_mb - reserved_memory_mb
        
        if usable_memory_mb <= 0:
            logger.warning("Kritisch wenig Speicher verfügbar")
            return 1  # Minimale Batch-Größe
        
        # Optimale Batch-Größe berechnen
        max_affordable_batch = int(usable_memory_mb / estimated_memory_per_item_mb)
        
        # Mit Sicherheitsfaktor
        safety_factor = 0.8  # 80% des verfügbaren Speichers verwenden
        safe_batch_size = int(max_affordable_batch * safety_factor)
        
        optimal_batch_size = min(batch_size, max(1, safe_batch_size))
        
        if optimal_batch_size < batch_size:
            logger.info(
                f"Batch-Größe von {batch_size} auf {optimal_batch_size} "
                f"reduziert aufgrund Memory-Constraints"
            )
        
        return optimal_batch_size
    
    def estimate_memory_for_generation(
        self,
        sequence_length: int,
        batch_size: int,
        model_params: int = None
    ) -> float:
        """
        Schätzt Memory-Verwendung für Text-Generierung
        
        Args:
            sequence_length: Länge der Sequenz in Tokens
            batch_size: Batch-Größe
            model_params: Anzahl Model-Parameter (optional)
            
        Returns:
            Geschätzte Memory-Verwendung in MB
        """
        
        # Basis-Schätzungen (können verfeinert werden)
        bytes_per_float16 = 2
        bytes_per_int32 = 4
        
        # Model Memory (falls nicht bereits geladen)
        model_memory_mb = 0
        if model_params:
            model_memory_mb = (model_params * bytes_per_float16) / (1024**2)
        
        # KV-Cache Memory
        # Vereinfachte Schätzung: head_dim * n_heads * 2 (K + V) * seq_len * batch_size
        estimated_head_dim = 64
        estimated_n_heads = 32
        kv_cache_memory_mb = (
            estimated_head_dim * estimated_n_heads * 2 * 
            sequence_length * batch_size * bytes_per_float16
        ) / (1024**2)
        
        # Activation Memory
        activation_memory_mb = (
            sequence_length * batch_size * 4096 * bytes_per_float16  # Geschätzte Hidden Size
        ) / (1024**2)
        
        # Tokenizer Memory
        tokenizer_memory_mb = sequence_length * batch_size * bytes_per_int32 / (1024**2)
        
        total_estimated_mb = (
            model_memory_mb + 
            kv_cache_memory_mb + 
            activation_memory_mb + 
            tokenizer_memory_mb
        )
        
        # Overhead-Faktor
        overhead_factor = 1.5
        total_with_overhead = total_estimated_mb * overhead_factor
        
        return total_with_overhead
    
    def estimate_memory_for_embedding(
        self,
        text_length_chars: int,
        batch_size: int,
        embedding_dim: int = 384
    ) -> float:
        """
        Schätzt Memory-Verwendung für Embedding-Generierung
        
        Args:
            text_length_chars: Durchschnittliche Textlänge in Zeichen
            batch_size: Batch-Größe
            embedding_dim: Embedding-Dimensionen
            
        Returns:
            Geschätzte Memory-Verwendung in MB
        """
        
        bytes_per_float16 = 2
        
        # Geschätzte Token-Anzahl (vereinfacht: chars / 4)
        estimated_tokens = text_length_chars // 4
        
        # Input Memory
        input_memory_mb = (estimated_tokens * batch_size * 4) / (1024**2)  # Int32 für Tokens
        
        # Model Forward Pass Memory
        hidden_size = embedding_dim * 4  # Geschätzte Hidden Layer Größe
        forward_memory_mb = (
            estimated_tokens * batch_size * hidden_size * bytes_per_float16
        ) / (1024**2)
        
        # Output Embedding Memory
        output_memory_mb = (
            batch_size * embedding_dim * bytes_per_float16
        ) / (1024**2)
        
        total_estimated_mb = input_memory_mb + forward_memory_mb + output_memory_mb
        
        # Overhead
        overhead_factor = 1.3
        total_with_overhead = total_estimated_mb * overhead_factor
        
        return total_with_overhead
    
    def trigger_cleanup(self) -> bool:
        """
        Triggert Memory-Cleanup wenn nötig
        
        Returns:
            True wenn Cleanup durchgeführt wurde
        """
        status, usage_percent = self.check_memory_pressure()
        
        if status in ["warning", "critical"]:
            logger.info(f"Memory Cleanup getriggert - Usage: {usage_percent:.1%}")
            
            try:
                # MLX Memory Cleanup
                if mx.metal.is_available():
                    mx.metal.clear_cache()
                
                # Python Garbage Collection
                import gc
                gc.collect()
                
                # System-spezifisches Cleanup
                if self.is_apple_silicon:
                    self._apple_silicon_cleanup()
                
                return True
                
            except Exception as e:
                logger.warning(f"Memory Cleanup fehlgeschlagen: {e}")
                return False
        
        return False
    
    def _apple_silicon_cleanup(self):
        """Apple Silicon spezifisches Memory-Cleanup"""
        try:
            # Unified Memory Optimierungen
            # Hier könnten spezifische Apple Silicon Memory-Management Techniken
            # implementiert werden
            pass
        except Exception as e:
            logger.debug(f"Apple Silicon Cleanup Fehler: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Gibt umfassende Memory-Statistiken zurück"""
        current_usage = self.get_current_usage()
        status, usage_percent = self.check_memory_pressure()
        
        # Trends berechnen (falls genügend Snapshots vorhanden)
        trends = self._calculate_trends()
        
        return {
            'current': current_usage,
            'status': status,
            'usage_percent': usage_percent,
            'available_mb': self.get_available_memory_mb(),
            'trends': trends,
            'apple_silicon': self.is_apple_silicon,
            'unified_memory': self.unified_memory,
            'mlx_baseline_mb': self.mlx_baseline,
            'snapshots_count': len(self.snapshots)
        }
    
    def _calculate_trends(self) -> Dict[str, float]:
        """Berechnet Memory-Usage Trends"""
        if len(self.snapshots) < 2:
            return {}
        
        try:
            recent_snapshots = self.snapshots[-10:]  # Letzte 10 Snapshots
            
            # System Memory Trend
            system_usage_trend = (
                recent_snapshots[-1].system_used_gb - recent_snapshots[0].system_used_gb
            ) / len(recent_snapshots)
            
            # MLX Memory Trend
            mlx_usage_trend = (
                recent_snapshots[-1].mlx_allocated_mb - recent_snapshots[0].mlx_allocated_mb
            ) / len(recent_snapshots)
            
            # Process Memory Trend
            process_trend = (
                recent_snapshots[-1].process_rss_mb - recent_snapshots[0].process_rss_mb
            ) / len(recent_snapshots)
            
            return {
                'system_gb_per_snapshot': system_usage_trend,
                'mlx_mb_per_snapshot': mlx_usage_trend,
                'process_mb_per_snapshot': process_trend
            }
            
        except Exception as e:
            logger.debug(f"Trend-Berechnung fehlgeschlagen: {e}")
            return {}
    
    def log_memory_status(self, level: str = "info"):
        """Loggt aktuellen Memory-Status"""
        stats = self.get_memory_stats()
        
        log_func = getattr(logger, level, logger.info)
        
        log_func(
            f"Memory Status: {stats['status']} ({stats['usage_percent']:.1%}) | "
            f"Available: {stats['available_mb']:.0f}MB | "
            f"MLX: {stats['current']['mlx_allocated_mb']:.1f}MB | "
            f"Process: {stats['current']['process_rss_mb']:.1f}MB"
        )
    
    def reset_peak_tracking(self):
        """Setzt Peak-Memory-Tracking zurück"""
        self.mlx_peak_usage = self._get_mlx_memory_mb()
        logger.info("Peak Memory Tracking zurückgesetzt")
    
    def export_memory_profile(self, filename: str):
        """Exportiert Memory-Profile für Analyse"""
        try:
            import json
            
            profile_data = {
                'config': {
                    'apple_silicon': self.is_apple_silicon,
                    'unified_memory': self.unified_memory,
                    'baseline_mb': self.mlx_baseline
                },
                'snapshots': [
                    {
                        'timestamp': s.timestamp,
                        'system_total_gb': s.system_total_gb,
                        'system_available_gb': s.system_available_gb,
                        'system_used_gb': s.system_used_gb,
                        'process_rss_mb': s.process_rss_mb,
                        'mlx_allocated_mb': s.mlx_allocated_mb,
                        'mlx_peak_mb': s.mlx_peak_mb
                    }
                    for s in self.snapshots
                ],
                'stats': self.get_memory_stats()
            }
            
            with open(filename, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            logger.info(f"Memory Profile exportiert nach: {filename}")
            
        except Exception as e:
            logger.error(f"Memory Profile Export fehlgeschlagen: {e}")


# Utility Functions
def get_system_memory_info() -> Dict[str, float]:
    """Gibt System Memory Informationen zurück"""
    memory = psutil.virtual_memory()
    
    return {
        'total_gb': memory.total / (1024**3),
        'available_gb': memory.available / (1024**3),
        'used_gb': (memory.total - memory.available) / (1024**3),
        'percent_used': memory.percent,
        'free_gb': memory.free / (1024**3),
        'cached_gb': getattr(memory, 'cached', 0) / (1024**3)
    }


def detect_memory_constraints() -> Dict[str, Any]:
    """Erkennt Memory-Constraints des Systems"""
    info = get_system_memory_info()
    
    constraints = {
        'low_memory': info['total_gb'] < 8,
        'very_low_memory': info['total_gb'] < 4,
        'memory_pressure': info['percent_used'] > 80,
        'critical_memory': info['percent_used'] > 90,
        'recommended_batch_size': 32,
        'max_safe_batch_size': 128
    }
    
    # Anpassungen basierend auf verfügbarem Speicher
    if constraints['low_memory']:
        constraints['recommended_batch_size'] = 8
        constraints['max_safe_batch_size'] = 16
    
    if constraints['very_low_memory']:
        constraints['recommended_batch_size'] = 2
        constraints['max_safe_batch_size'] = 4
    
    return constraints