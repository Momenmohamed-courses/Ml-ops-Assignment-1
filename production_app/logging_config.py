import logging

from production_app.config import settings

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logging(level: str = settings.log_level)->None:
     logging.basicConfig(level = level, format = LOG_FORMAT, datefmt = DATE_FORMAT)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)