import logging
import os
from datetime import datetime

def setup_logger(log_dir="logs"):
    
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a timestamped log file
    log_file = os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Configure the logger
    logger = logging.getLogger("ImageCaptioningLogger")
    logger.setLevel(logging.INFO)

    # Create a file handler and set the formatter
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger

def log_metrics(logger, epoch, loss, val_loss, bleu_score=None):
    
    logger.info(f"Epoch {epoch + 1}")
    logger.info(f"Training Loss: {loss:.4f}")
    logger.info(f"Validation Loss: {val_loss:.4f}")
    if bleu_score is not None:
        logger.info(f"BLEU Score: {bleu_score:.4f}")
    logger.info("-" * 50)