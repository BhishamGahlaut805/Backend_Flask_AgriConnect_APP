"""
Centralized Logging Configuration
Provides unified logging across all modules with MongoDB storage support
"""

import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import traceback
import os

# ============================================================
# CUSTOM HANDLERS
# ============================================================

class MongoDBLogHandler(logging.Handler):
    """
    Custom logging handler that sends logs to MongoDB
    Stores logs in 'system_logs' collection for persistence
    """

    def __init__(self, level=logging.WARNING):
        super().__init__(level)
        self.mongo_storage = None
        self._init_mongo()

    def _init_mongo(self):
        """Initialize MongoDB connection lazily"""
        try:
            from mongo_storage import mongo_storage
            self.mongo_storage = mongo_storage
        except ImportError:
            # Fallback if mongo_storage not available
            print("Warning: mongo_storage not available for logging")
            self.mongo_storage = None

    def emit(self, record):
        """Send log record to MongoDB"""
        if self.mongo_storage and self.mongo_storage.db is not None:
            try:
                log_entry = {
                    "level": record.levelname,
                    "message": self.format(record),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                    "process": record.process,
                    "thread": record.thread,
                    "timestamp": datetime.utcnow(),
                    "logger_name": record.name
                }

                # Add exception info if present
                if record.exc_info:
                    log_entry["exception"] = {
                        "type": record.exc_info[0].__name__,
                        "message": str(record.exc_info[1]),
                        "traceback": traceback.format_exception(*record.exc_info)
                    }

                # Add extra fields if present
                if hasattr(record, 'extra_data'):
                    log_entry["extra_data"] = record.extra_data

                self.mongo_storage.logs.insert_one(log_entry)
            except Exception as e:
                # Don't let logging errors crash the application
                print(f"Failed to write log to MongoDB: {e}")


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    Useful for log aggregation and analysis
    """

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage()
        }

        # Add exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add extra fields
        if hasattr(record, 'extra_data'):
            log_entry["extra_data"] = record.extra_data

        return json.dumps(log_entry)


class ColoredConsoleFormatter(logging.Formatter):
    """
    Custom formatter with colors for console output
    Makes logs more readable during development
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        # Format timestamp
        record.asctime = self.formatTime(record, self.datefmt)

        return super().format(record)


# ============================================================
# LOGGER CONFIGURATION
# ============================================================

def setup_logging(log_level: str = None, enable_mongo: bool = True,
                  enable_file: bool = True, log_dir: str = "logs"):
    """
    Setup centralized logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_mongo: Enable MongoDB logging (for warnings and above)
        enable_file: Enable file logging
        log_dir: Directory for log files

    Returns:
        Configured root logger
    """

    # Set default log level from environment or INFO
    if log_level is None:
        log_level = os.getenv("LOG_LEVEL", "INFO")

    log_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logs directory if needed
    if enable_file and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplication
    root_logger.handlers.clear()

    # ============================================================
    # CONSOLE HANDLER (Always enabled)
    # ============================================================
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    # Colored console formatter
    console_formatter = ColoredConsoleFormatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ============================================================
    # FILE HANDLER (Rotating files)
    # ============================================================
    if enable_file:
        # Regular log file - rotates at midnight
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, "app.log"),
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        # Error log file - only errors and above
        error_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, "error.log"),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s\n%(exc_info)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        root_logger.addHandler(error_handler)

        # JSON log file for structured logging
        json_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, "app.json.log"),
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8"
        )
        json_handler.setLevel(logging.DEBUG)
        json_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(json_handler)

    # ============================================================
    # MONGODB HANDLER (For persistent storage)
    # ============================================================
    if enable_mongo:
        try:
            mongo_handler = MongoDBLogHandler(level=logging.WARNING)
            mongo_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            mongo_handler.setFormatter(mongo_formatter)
            root_logger.addHandler(mongo_handler)
            root_logger.info("MongoDB logging handler initialized")
        except Exception as e:
            root_logger.warning(f"Failed to initialize MongoDB logging: {e}")

    # Log startup message
    root_logger.info(f"Logging initialized | Level: {logging.getLevelName(log_level)} | "
                    f"File logging: {enable_file} | MongoDB logging: {enable_mongo}")

    return root_logger


# ============================================================
# LOGGER FACTORY
# ============================================================

def get_logger(name: str, level: str = None) -> logging.Logger:
    """
    Get a configured logger instance for a module

    Args:
        name: Logger name (typically __name__ from the calling module)
        level: Optional specific log level for this logger

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)

    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    return logger


class LoggerContext:
    """
    Context manager for adding extra context to logs
    Usage:
        with LoggerContext(logger, user_id="123", request_id="abc"):
            logger.info("Processing request")
    """

    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.extra = kwargs
        self.old_factory = None

    def __enter__(self):
        # Store old factory
        self.old_factory = logging.getLogRecordFactory()

        # Get current factory or default
        factory = self.old_factory or logging.LogRecord

        def record_factory(*args, **kwargs):
            record = factory(*args, **kwargs)
            for key, value in self.extra.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


# ============================================================
# PERFORMANCE LOGGING
# ============================================================

class PerformanceLogger:
    """
    Helper class for logging performance metrics
    """

    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.utcnow()
        self.logger.debug(f"Starting: {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(f"Failed: {self.operation} | Duration: {duration:.3f}s | Error: {exc_val}")
        else:
            self.logger.info(f"Completed: {self.operation} | Duration: {duration:.3f}s")

    def checkpoint(self, checkpoint_name: str):
        """Log a checkpoint during the operation"""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        self.logger.debug(f"Checkpoint '{checkpoint_name}' at {elapsed:.3f}s")


def log_performance(logger: logging.Logger, operation: str):
    """Decorator for logging function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceLogger(logger, f"{operation} - {func.__name__}"):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ============================================================
# REQUEST LOGGING (For Flask)
# ============================================================

class RequestLogger:
    """
    Helper for logging HTTP requests with detailed information
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_request(self, request, response=None, duration_ms: float = None):
        """Log HTTP request details"""
        log_data = {
            "method": request.method,
            "path": request.path,
            "remote_addr": request.remote_addr,
            "user_agent": request.headers.get("User-Agent", "Unknown"),
            "status_code": response.status_code if response else None,
            "duration_ms": duration_ms
        }

        self.logger.info(f"Request: {log_data['method']} {log_data['path']} | "
                        f"Status: {log_data['status_code']} | "
                        f"Duration: {duration_ms:.2f}ms" if duration_ms else "")

        # Log warnings for slow requests
        if duration_ms and duration_ms > 1000:
            self.logger.warning(f"Slow request: {log_data['method']} {log_data['path']} took {duration_ms:.2f}ms")

        return log_data


# ============================================================
# INITIALIZE DEFAULT LOGGER
# ============================================================

# Setup default logging configuration
logger = setup_logging()

# Export commonly used functions
__all__ = [
    'setup_logging',
    'get_logger',
    'LoggerContext',
    'PerformanceLogger',
    'log_performance',
    'RequestLogger',
    'logger'
]
