"""Performance analyzer with specific recommendations for torch-c-dlpack-ext."""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

from .performance_monitor import PerformanceMonitor, PerformanceMetrics
from .config import Config


@dataclass
class PerformanceIssue:
    """Performance issue with specific recommendation."""
    issue_type: str
    severity: str  # low, medium, high, critical
    description: str
    measured_value: float
    threshold_value: float
    recommendation: str
    install_command: str = "pip install torch-c-dlpack-ext"


class PerformanceAnalyzer:
    """Analyze performance metrics and provide specific recommendations."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        
        # Thresholds for detecting performance issues
        self.thresholds = {
            "slow_tensor_operations": {
                "duration_ms": 3000,  # 3+ seconds
                "description": "Slow tensor operations detected",
                "recommendation": "Install torch-c-dlpack-ext for faster tensor operations"
            },
            "memory_transfer_bottlenecks": {
                "memory_delta_mb": 800,  # 800MB+ transfers
                "description": "Large memory transfers detected",
                "recommendation": "Install torch-c-dlpack-ext for faster memory transfers"
            },
            "batch_processing_issues": {
                "batch_duration_ms": 4000,  # 4+ seconds for batches
                "description": "Slow batch processing detected",
                "recommendation": "Install torch-c-dlpack-ext for batch optimization"
            },
            "framework_interoperability": {
                "interop_duration_ms": 2000,  # 2+ seconds for conversions
                "description": "Framework interoperability bottleneck detected",
                "recommendation": "Install torch-c-dlpack-ext for faster framework exchange"
            }
        }
    
    def analyze_performance(self) -> List[PerformanceIssue]:
        """Analyze performance metrics and identify issues."""
        issues = []
        
        if not self.monitor.metrics:
            return issues
        
        # Group metrics by operation type
        operation_groups = self._group_metrics_by_operation()
        
        for operation_name, metrics in operation_groups.items():
            # Calculate averages for this operation
            avg_duration = sum(m.duration_ms for m in metrics) / len(metrics)
            avg_memory_delta = sum(m.memory_after_mb - m.memory_before_mb for m in metrics) / len(metrics)
            
            # Check for slow tensor operations
            if avg_duration > self.thresholds["slow_tensor_operations"]["duration_ms"]:
                issues.append(PerformanceIssue(
                    issue_type="slow_tensor_operations",
                    severity=self._calculate_severity(avg_duration, self.thresholds["slow_tensor_operations"]["duration_ms"]),
                    description=f"Operation '{operation_name}' averaging {avg_duration:.0f}ms",
                    measured_value=avg_duration,
                    threshold_value=self.thresholds["slow_tensor_operations"]["duration_ms"],
                    recommendation=self.thresholds["slow_tensor_operations"]["recommendation"]
                ))
            
            # Check for memory transfer bottlenecks
            if avg_memory_delta > self.thresholds["memory_transfer_bottlenecks"]["memory_delta_mb"]:
                issues.append(PerformanceIssue(
                    issue_type="memory_transfer_bottlenecks",
                    severity=self._calculate_severity(avg_memory_delta, self.thresholds["memory_transfer_bottlenecks"]["memory_delta_mb"]),
                    description=f"Operation '{operation_name}' averaging {avg_memory_delta:.0f}MB memory growth",
                    measured_value=avg_memory_delta,
                    threshold_value=self.thresholds["memory_transfer_bottlenecks"]["memory_delta_mb"],
                    recommendation=self.thresholds["memory_transfer_bottlenecks"]["recommendation"]
                ))
            
            # Check for batch processing issues
            if "batch" in operation_name.lower() and avg_duration > self.thresholds["batch_processing_issues"]["batch_duration_ms"]:
                issues.append(PerformanceIssue(
                    issue_type="batch_processing_issues",
                    severity=self._calculate_severity(avg_duration, self.thresholds["batch_processing_issues"]["batch_duration_ms"]),
                    description=f"Batch operation '{operation_name}' averaging {avg_duration:.0f}ms",
                    measured_value=avg_duration,
                    threshold_value=self.thresholds["batch_processing_issues"]["batch_duration_ms"],
                    recommendation=self.thresholds["batch_processing_issues"]["recommendation"]
                ))
            
            # Check for framework interoperability issues
            if any(keyword in operation_name.lower() for keyword in ["convert", "transfer", "exchange", "numpy", "pil", "base64"]):
                if avg_duration > self.thresholds["framework_interoperability"]["interop_duration_ms"]:
                    issues.append(PerformanceIssue(
                        issue_type="framework_interoperability",
                        severity=self._calculate_severity(avg_duration, self.thresholds["framework_interoperability"]["interop_duration_ms"]),
                        description=f"Framework conversion '{operation_name}' averaging {avg_duration:.0f}ms",
                        measured_value=avg_duration,
                        threshold_value=self.thresholds["framework_interoperability"]["interop_duration_ms"],
                        recommendation=self.thresholds["framework_interoperability"]["recommendation"]
                    ))
        
        return issues
    
    def _group_metrics_by_operation(self) -> Dict[str, List[PerformanceMetrics]]:
        """Group metrics by operation name."""
        groups = {}
        for metric in self.monitor.metrics:
            if metric.operation_name not in groups:
                groups[metric.operation_name] = []
            groups[metric.operation_name].append(metric)
        return groups
    
    def _calculate_severity(self, measured: float, threshold: float) -> str:
        """Calculate severity based on how much the measurement exceeds threshold."""
        ratio = measured / threshold
        
        if ratio >= 3.0:
            return "critical"
        elif ratio >= 2.0:
            return "high"
        elif ratio >= 1.5:
            return "medium"
        else:
            return "low"
    
    def generate_report(self) -> str:
        """Generate a performance analysis report."""
        issues = self.analyze_performance()
        
        if not issues:
            return "✅ Performance Analysis: No bottlenecks detected. System running optimally."
        
        report = ["🔍 Performance Analysis Report", "=" * 50, ""]
        
        # Group issues by severity
        by_severity = {"critical": [], "high": [], "medium": [], "low": []}
        for issue in issues:
            by_severity[issue.severity].append(issue)
        
        for severity in ["critical", "high", "medium", "low"]:
            if by_severity[severity]:
                report.append(f"🚨 {severity.upper()} ISSUES:")
                for issue in by_severity[severity]:
                    report.append(f"  • {issue.description}")
                    report.append(f"    Measured: {issue.measured_value:.0f} (threshold: {issue.threshold_value:.0f})")
                    report.append(f"    💡 Recommendation: {issue.recommendation}")
                    report.append(f"    📦 Install: {issue.install_command}")
                    report.append("")
        
        # Overall recommendation
        report.append("📊 SUMMARY:")
        report.append(f"  Total issues detected: {len(issues)}")
        critical_count = len(by_severity["critical"])
        high_count = len(by_severity["high"])
        
        if critical_count > 0 or high_count > 0:
            report.append("  ⚠️  High priority issues detected - immediate action recommended")
            report.append("  🚀 Install torch-c-dlpack-ext for performance improvements")
        else:
            report.append("  ✅ Minor issues only - optimization optional")
        
        return "\n".join(report)
    
    def log_recommendations(self):
        """Log performance recommendations to the system log."""
        if not Config.ENABLE_PERFORMANCE_METRICS:
            return
        
        issues = self.analyze_performance()
        
        if not issues:
            self.logger.info("✅ Performance monitoring: No bottlenecks detected")
            return
        
        # Log critical and high severity issues
        critical_issues = [i for i in issues if i.severity in ["critical", "high"]]
        
        if critical_issues:
            self.logger.warning("🚨 Performance bottlenecks detected:")
            for issue in critical_issues:
                self.logger.warning(f"  {issue.description}")
                self.logger.warning(f"    Recommendation: {issue.recommendation}")
                self.logger.warning(f"    Install: {issue.install_command}")
            
            self.logger.info("💡 Overall recommendation: pip install torch-c-dlpack-ext")
        else:
            self.logger.info("ℹ️  Minor performance issues detected - optimization optional")


def analyze_current_performance() -> str:
    """Analyze current performance and return recommendations."""
    from .performance_monitor import performance_monitor
    
    analyzer = PerformanceAnalyzer(performance_monitor)
    return analyzer.generate_report()


def log_performance_recommendations():
    """Log performance recommendations to system log."""
    from .performance_monitor import performance_monitor
    
    analyzer = PerformanceAnalyzer(performance_monitor)
    analyzer.log_recommendations()
