"""
YiRage Backend API

This module provides Python interfaces for querying and selecting backends.
"""

import sys
from typing import List, Optional, Dict, Any

# Try to import the core module
try:
    from . import core
    HAS_CORE = True
except ImportError:
    HAS_CORE = False


def get_available_backends() -> List[str]:
    """
    Get list of available backend names.
    
    Returns:
        List of backend names (e.g., ['cuda', 'cpu', 'mps', 'maca', 'ascend'])
    
    Example:
        >>> import yirage
        >>> backends = yirage.get_available_backends()
        >>> print(f"Available backends: {backends}")
        Available backends: ['cpu', 'mps', 'maca']
    """
    if not HAS_CORE:
        return []
    
    try:
        # Call C++ function if available
        if hasattr(core, 'get_available_backends'):
            return core.get_available_backends()
    except Exception as e:
        print(f"Warning: Could not query C++ backends: {e}", file=sys.stderr)
    
    return []


def is_backend_available(backend: str) -> bool:
    """
    Check if a specific backend is available.
    
    Args:
        backend: Backend name (e.g., 'cuda', 'cpu', 'mps', 'maca', 'ascend')
    
    Returns:
        True if backend is available, False otherwise
    
    Example:
        >>> import yirage
        >>> if yirage.is_backend_available('maca'):
        ...     print("MACA backend is available")
        MACA backend is available
    """
    return backend in get_available_backends()


def get_default_backend() -> Optional[str]:
    """
    Get the default backend name.
    
    Returns:
        Default backend name, or None if no backends available
    
    Example:
        >>> import yirage
        >>> backend = yirage.get_default_backend()
        >>> print(f"Default backend: {backend}")
        Default backend: cuda
    """
    backends = get_available_backends()
    if not backends:
        return None
    
    # Priority order: CUDA > MACA > Ascend > MPS > CPU
    priority = ['cuda', 'maca', 'ascend', 'mps', 'cpu']
    for backend in priority:
        if backend in backends:
            return backend
    
    # Otherwise return first available
    return backends[0]


def get_backend_info(backend: str) -> Dict[str, Any]:
    """
    Get detailed information about a backend.
    
    Args:
        backend: Backend name
    
    Returns:
        Dictionary with backend information
    """
    # Simple implementation - just return name and availability
    return {
        'name': backend,
        'available': is_backend_available(backend),
    }


def set_default_backend(backend: str) -> bool:
    """
    Set the default backend.
    
    Args:
        backend: Backend name to set as default ('cuda', 'maca', 'mps', 'cpu', 'ascend')
    
    Returns:
        True if successful, False if backend not available
    
    Example:
        >>> import yirage
        >>> success = yirage.set_default_backend('maca')
        >>> print(f"Set default backend: {success}")
        Set default backend: True
    """
    if not is_backend_available(backend):
        return False
    
    try:
        if HAS_CORE and hasattr(core, 'set_default_backend'):
            return core.set_default_backend(backend)
    except:
        pass
    
    return False


def list_backends(verbose: bool = False) -> None:
    """Print available backends to stdout."""
    backends = get_available_backends()
    
    if not backends:
        print("No backends available")
        return
    
    default = get_default_backend()
    
    print("Available YiRage Backends:")
    for backend_name in backends:
        marker = " [default]" if backend_name == default else ""
        print(f"  - {backend_name}{marker}")


# Convenience aliases
available_backends = get_available_backends
default_backend = get_default_backend

