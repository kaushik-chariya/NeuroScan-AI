import sys
import traceback
from logger import logger


def get_error_details(error: Exception) -> str:
    """
    Extracts detailed error information including
    filename, line number, and error message.
    """
    _, _, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = str(error)

    detailed_message = (
        f"\n"
        f"{'='*60}\n"
        f"📁 File     : {file_name}\n"
        f"📍 Line     : {line_number}\n"
        f"❌ Error    : {error_message}\n"
        f"{'='*60}"
    )
    return detailed_message


class NeuroScanException(Exception):
    """
    Custom Exception class for NeuroScan-AI project.
    Logs error details automatically on raise.
    """

    def __init__(self, error: Exception, error_detail: sys):
        super().__init__(str(error))

        self.error_message = get_error_details(error)
        logger.error(f"🚨 Exception Caught! {self.error_message}")

    def __str__(self):
        return self.error_message


# ── Test ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        logger.info("🧪 Testing NeuroScanException...")

        # Simulating a file not found error
        with open("non_existent_file.txt", "r") as f:
            data = f.read()

    except Exception as e:
        raise NeuroScanException(e, sys)

    try:
        logger.info("🧪 Testing division by zero...")
        result = 10 / 0

    except Exception as e:
        raise NeuroScanException(e, sys)