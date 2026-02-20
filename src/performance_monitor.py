"""Performance monitoring for tensor operations and memory usage."""

import time
import psutil
import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from .config import Config


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    gpu_memory_before_mb: Optional[float] = None
    gpu_memory_after_mb: Optional[float] = None
    tensor_count_before: Optional[int] = None
    tensor_count_after: Optional[int] = None


class PerformanceMonitor:
    """Monitor performance and detect bottlenecks."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        
    def start_operation(self, operation_name: str) -> float:
        """Start monitoring an operation."""
        return time.time()
    
    def end_operation(self, operation_name: str, start_time: float) -> PerformanceMetrics:
        """End monitoring and record metrics."""
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # System memory
        memory_before = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_after = self.process.memory_info().rss / 1024 / 1024   # MB
        memory_peak = self.process.memory_info().peak_wset / 1024 / 1024  # MB
        
        # GPU memory if available
        gpu_memory_before = None
        gpu_memory_after = None
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            gpu_memory_after = torch.cuda.memory_allocated() / 1024 / 1024   # MB
        
        # Tensor count if available
        tensor_count_before = None
        tensor_count_after = None
        if torch.cuda.is_available():
            tensor_count_before = len(torch.cuda.memory_allocated())
            tensor_count_after = len(torch.cuda.memory_allocated())
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_peak_mb=memory_peak,
            gpu_memory_before_mb=gpu_memory_before,
            gpu_memory_after_mb=gpu_memory_after,
            tensor_count_before=tensor_count_before,
            tensor_count_after=tensor_count_after
        )
        
        self.metrics.append(metrics)
        return metrics
    
    def analyze_bottlenecks(self) -> Dict[str, List[str]]:
        """Analyze metrics and identify performance bottlenecks."""
        recommendations = {}
        
        if not self.metrics:
            return recommendations
        
        # Group metrics by operation
        operation_metrics = {}
        for metric in self.metrics:
            if metric.operation_name not in operation_metrics:
                operation_metrics[metric.operation_name] = []
            operation_metrics[metric.operation_name].append(metric)
        
        # Analyze each operation type
        for operation, metrics_list in operation_metrics.items():
            operation_recommendations = []
            
            # Calculate averages
            avg_duration = sum(m.duration_ms for m in metrics_list) / len(metrics_list)
            avg_memory_delta = sum(m.memory_after_mb - m.memory_before_mb for m in metrics_list) / len(metrics_list)
            
            # GPU memory analysis
            gpu_metrics = [m for m in metrics_list if m.gpu_memory_before_mb is not None]
            if gpu_metrics:
                avg_gpu_delta = sum(m.gpu_memory_after_mb - m.gpu_memory_before_mb for m in gpu_metrics) / len(gpu_metrics)
                max_gpu_memory = max(m.gpu_memory_after_mb for m in gpu_metrics)
                
                # Memory transfer bottlenecks
                if avg_gpu_delta > 1000:  # 1GB+ transfers
                    operation_recommendations.append(
                        f"⚠️  Large GPU memory transfers detected ({avg_gpu_delta:.0f}MB avg). "
                        f"Recommend: pip install torch-c-dlpack-ext for faster tensor exchange"
                    )
                
                # High GPU memory usage
                if max_gpu_memory > 10000:  # 10GB+
                    operation_recommendations.append(
                        f"⚠️  High GPU memory usage ({max_gpu_memory:.0f}MB). "
                        f"Recommend: pip install torch-c-dlpack-ext for memory optimization"
                    )
            
            # Slow operations
            if avg_duration > 5000:  # 5+ seconds
                operation_recommendations.append(
                    f"⚠️  Slow operation detected ({avg_duration:.0f}ms avg). "
                    f"Recommend: pip install torch-c-dlpack-ext for tensor acceleration"
                )
            
            # Memory growth
            if avg_memory_delta > 500:  # 500MB+ growth
                operation_recommendations.append(
                    f"⚠️  High memory growth detected ({avg_memory_delta:.0f}MB avg). "
                    f"Recommend: pip install torch-c-dlpack-ext for memory efficiency"
                )
            
            # Batch processing issues
            if "batch" in operation.lower() and avg_duration > 2000:
                operation_recommendations.append(
                    f"⚠️  Slow batch processing ({avg_duration:.0f}ms avg). "
                    f"Recommend: pip install torch-c-dlpack-ext for batch optimization"
                )
            
            # Framework interoperability (multiple frameworks)
            if any(keyword in operation.lower() for keyword in ["convert", "transfer", "exchange", "numpy", "pil"]):
                if avg_duration > 1000:
                    operation_recommendations.append(
                        f"⚠️  Framework interoperability bottleneck ({avg_duration:.0f}ms avg). "
                        f"Recommend: pip install torch-c-dlpack-ext for faster framework exchange"
                    )
            
            if operation_recommendations:
                recommendations[operation] = operation_recommendations
        
        return recommendations
    
    def log_performance_summary(self):
        """Log performance summary with recommendations."""
        if not Config.ENABLE_PERFORMANCE_METRICS:
            return
        
        recommendations = self.analyze_bottlenecks()
        
        if not recommendations:
            self.logger.info("✅ Performance monitoring: No bottlenecks detected")
            return
        
        self.logger.warning("🔍 Performance monitoring: Bottlenecks detected")
        
        for operation, recs in recommendations.items():
            self.logger.warning(f"Operation: {operation}")
            for rec in recs:
                self.logger.warning(f"  {rec}")
        
        # Overall recommendation
        self.logger.info("💡 Overall recommendation: pip install torch-c-dlpack-ext for performance improvements")
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """Get summary statistics of all metrics."""
        if not self.metrics:
            return {}
        
        return {
            "total_operations": len(self.metrics),
            "avg_duration_ms": sum(m.duration_ms for m in self.metrics) / len(self.metrics),
            "max_duration_ms": max(m.duration_ms for m in self.metrics),
            "total_memory_growth_mb": sum(m.memory_after_mb - m.memory_before_mb for m in self.metrics),
            "avg_memory_growth_mb": sum(m.memory_after_mb - m.memory_before_mb for m in self.metrics) / len(self.metrics)
        }


# Global instance
performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = performance_monitor.start_operation(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                performance_monitor.end_operation(operation_name, start_time)
        return wrapper
    return decorator
