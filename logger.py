import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Log directory setup
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log")


# Custom Formatter with Emojis
class EmojiFormatter(logging.Formatter):
    EMOJIS = {
        logging.DEBUG:    "🔍 DEBUG",
        logging.INFO:     "✅ INFO",
        logging.WARNING:  "⚠️  WARNING",
        logging.ERROR:    "❌ ERROR",
        logging.CRITICAL: "🔥 CRITICAL",
    }

    FORMAT = "[%(asctime)s] | {level} | %(module)s | %(funcName)s | line:%(lineno)d | %(message)s"

    def format(self, record):
        level = self.EMOJIS.get(record.levelno, "📝 LOG")
        formatter = logging.Formatter(
            self.FORMAT.format(level=level),
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        return formatter.format(record)


def get_logger(name: str = "NeuroScan") -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)

    # ── Console Handler ──────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(EmojiFormatter())

    # ── File Handler (Rotating — max 5MB, 3 backups) ─
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] | %(levelname)s | %(module)s | %(funcName)s | line:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = get_logger("NeuroScan")


# ── Test ─────────────────────────────────────────────
if __name__ == "__main__":
    logger.debug("Debugging started 🧪")
    logger.info("Data pipeline is starting...")
    logger.warning("GPU not found, running on CPU")
    logger.error("Model failed to load!")
    logger.critical("System crashed 💀")