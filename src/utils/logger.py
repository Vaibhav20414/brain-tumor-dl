import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with console (and optional file) handler.

    Args:
        name:     Logger name (typically __name__).
        log_file: Optional path to write logs to a file.
        level:    Logging level.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Force UTF-8 on Windows consoles (cp1252 can't encode unicode symbols)
    stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1) if sys.platform == "win32" else sys.stdout
    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
