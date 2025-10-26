# monitor.py
from loguru import logger
import os

os.makedirs("logs", exist_ok=True)
logger.add("logs/app.log", rotation="1 MB")

def log_event(event_type: str, message: str):
    logger.info(f"[{event_type.upper()}] {message}")