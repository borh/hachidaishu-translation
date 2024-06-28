from loguru import logger
import inspect
import logging
import sys

logger.add(sys.stderr, format="{time} {level} {message}", filter="", level="INFO")
logger.add("run_{time}.log", level="DEBUG")


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
# import logging
# from datetime import datetime
#
# # Set the root logger to a higher level to prevent it from handling lower-level logs
# logging.basicConfig(level=logging.DEBUG)
# logging.getLogger().setLevel(logging.WARNING)
#
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
#
# # Remove all handlers associated with the root logger object
# for handler in logging.root.handlers[:]:
#     logging.root.removeHandler(handler)
#
# # Create a console handler and set the level to info
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.WARNING)
#
# # Create a file handler and set the level to info
# log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
# file_handler = logging.FileHandler(log_filename, mode="w")
# file_handler.setLevel(logging.DEBUG)  # DEBUG will give us OpenAI API response logs
#
# # Create a formatter and set it for both handlers
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)
#
# # Add the handlers to the logger
# logger.addHandler(console_handler)
# logger.addHandler(file_handler)
