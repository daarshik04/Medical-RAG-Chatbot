import sys
import traceback


class CustomException(Exception):
    """
    FIX 8: The original implementation called sys.exc_info() inside __init__,
    which is fine when raised inside an except block but returns (None, None, None)
    everywhere else — causing an AttributeError on exc_tb.tb_frame.

    Fix: guard against None traceback, and also capture a formatted traceback
    string so the full stack is always preserved in the log message.
    """

    def __init__(self, message: str, error_detail: Exception = None):
        self.error_message = self._build_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def _build_message(message: str, error_detail: Exception) -> str:
        # sys.exc_info() is only populated when we're inside an except block.
        _, _, exc_tb = sys.exc_info()

        if exc_tb is not None:
            file_name   = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
        else:
            # Called outside an except block — degrade gracefully
            file_name   = "Unknown File"
            line_number = "Unknown Line"

        base = (
            f"{message} | "
            f"Error: {error_detail} | "
            f"File: {file_name} | "
            f"Line: {line_number}"
        )

        # Append full traceback when one is available
        if error_detail is not None:
            tb_str = "".join(traceback.format_exception(type(error_detail), error_detail, exc_tb))
            base += f"\nTraceback:\n{tb_str}"

        return base

    def __str__(self) -> str:
        return self.error_message
