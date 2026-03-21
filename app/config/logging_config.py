import os
import logging
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

ENV = os.getenv("ENV")

def setup_logger(name="app", log_file="logs/app.log", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    if not logger.handlers:
        if ENV == "DEV":
            handler = logging.StreamHandler()
        else:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            handler = RotatingFileHandler(
                log_file, maxBytes=5 * 1024 * 1024, backupCount=5
            )

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger
