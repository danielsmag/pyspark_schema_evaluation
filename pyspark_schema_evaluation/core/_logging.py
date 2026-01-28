from __future__ import annotations

import logging
from typing import Optional

_logger: Optional[logging.Logger] = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get or create the module-level logger for schema evolution.
    
    This logger is configured with NullHandler by default, making it silent
    unless the user configures logging in their application.
    
    Args:
        name: Optional logger name. If None, uses 'spark_schema_evolution'
        
    Returns:
        Configured logger instance with NullHandler
        
    Example:
        >>> from spark_schema_evolution.utils._logging import get_logger
        >>> logger = get_logger()
        >>> logger.info("This is silent unless user configures logging")
    """
    global _logger
    
    if _logger is None:
        logger_name = name or "spark_schema_evolution"
        _logger = logging.getLogger(logger_name)
        
        # Add NullHandler to prevent "No handler found" warnings
        # This makes the library silent by default (pandas pattern)
        if not _logger.handlers:
            _logger.addHandler(logging.NullHandler())
    
    return _logger


def safe_log(logger: Optional[logging.Logger], level: str, message: str, **kwargs) -> None:
    """
    Safely log a message, handling any logger type or None.
    
    This function never raises exceptions - it silently fails if logging fails.
    This ensures the library never breaks due to logging issues.
    
    Args:
        logger: Logger instance (any type) or None
        level: Log level ('info', 'warning', 'error', 'debug')
        message: Message to log
        **kwargs: Additional arguments for logging
        
    Example:
        >>> from spark_schema_evolution.utils._logging import safe_log, get_logger
        >>> logger = get_logger()
        >>> safe_log(logger, 'info', 'Schema evolution started')
    """
    if logger is None:
        return
    
    try:
        if isinstance(logger, logging.Logger):
            log_method = getattr(logger, level, logger.info)
            if kwargs:
                log_method(message, **kwargs)
            else:
                log_method(message)
        elif hasattr(logger, level):
            getattr(logger, level)(message)
        elif hasattr(logger, 'log'):
            logger.log(message)
    except Exception:
        pass


_module_logger = get_logger()
