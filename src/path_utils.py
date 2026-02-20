"""Cross-platform path utilities for Windows and WSL compatibility."""

import os
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Union


class PathManager:
    """Manages cross-platform path resolution for Windows/WSL environments."""
    
    @staticmethod
    def normalize_path(path: Union[str, Path]) -> Path:
        """
        Normalize a path for the current platform.
        
        Args:
            path: Path to normalize (can be string or Path object)
            
        Returns:
            Normalized Path object for current platform
        """
        if isinstance(path, str):
            path = Path(path)
        
        # Convert to absolute path
        abs_path = path.resolve()
        
        # Handle WSL path conversion if needed
        if PathManager._is_wsl_environment() and PathManager._has_windows_path(abs_path):
            return PathManager._windows_to_wsl_path(abs_path)
        elif not PathManager._is_wsl_environment() and PathManager._has_wsl_path(abs_path):
            return PathManager._wsl_to_windows_path(abs_path)
        elif not PathManager._is_wsl_environment() and PathManager._has_windows_path(abs_path):
            # In Windows environment, ensure Windows path format
            return abs_path
        elif not PathManager._is_wsl_environment():
            # In Windows environment, if path doesn't exist, try relative to project root
            if not abs_path.exists():
                project_root = PathManager.get_project_root()
                relative_path = project_root / path
                if relative_path.exists():
                    return relative_path
        
        return abs_path
    
    @staticmethod
    def _is_wsl_environment() -> bool:
        """Check if running in WSL environment."""
        try:
            return 'microsoft' in os.uname().release.lower()
        except (AttributeError, OSError):
            return False
    
    @staticmethod
    def _has_windows_path(path: Path) -> bool:
        """Check if path contains Windows drive format."""
        return len(str(path)) > 1 and str(path)[1] == ':'
    
    @staticmethod
    def _has_wsl_path(path: Path) -> bool:
        """Check if path contains WSL mount format."""
        return '/mnt/' in str(path)
    
    @staticmethod
    def _windows_to_wsl_path(windows_path: Path) -> Path:
        """Convert Windows path to WSL path."""
        path_str = str(windows_path)
        if len(path_str) > 1 and path_str[1] == ':':
            drive = path_str[0].lower()
            rest_path = path_str[2:].replace('\\', '/')
            return Path(f"/mnt/{drive}{rest_path}")
        return windows_path
    
    @staticmethod
    def _wsl_to_windows_path(wsl_path: Path) -> Path:
        """Convert WSL path to Windows path."""
        path_str = str(wsl_path)
        if path_str.startswith('/mnt/'):
            parts = path_str[5:].split('/', 1)
            if len(parts) >= 2:
                drive = parts[0].upper()
                rest_path = parts[1].replace('/', '\\')
                return Path(f"{drive}:\\{rest_path}")
        return wsl_path
    
    @staticmethod
    def get_project_root() -> Path:
        """Get project root directory, cross-platform."""
        # Start from current file and go up to find project root
        current = Path(__file__).resolve()
        while current.parent != current:
            if (current / '.git').exists() or (current / 'src').exists():
                return current
            current = current.parent
        return current
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, creating if necessary.
        
        Args:
            path: Directory path to ensure
            
        Returns:
            Normalized Path object
        """
        normalized_path = PathManager.normalize_path(path)
        normalized_path.mkdir(parents=True, exist_ok=True)
        return normalized_path
    
    @staticmethod
    def safe_join(base: Union[str, Path], *paths: Union[str, Path]) -> Path:
        """
        Safely join paths with cross-platform normalization.
        
        Args:
            base: Base path
            *paths: Additional path components
            
        Returns:
            Normalized joined Path
        """
        base_path = PathManager.normalize_path(base)
        result = base_path
        for path in paths:
            result = result / path
        return PathManager.normalize_path(result)


# Global path manager instance
path_manager = PathManager()
