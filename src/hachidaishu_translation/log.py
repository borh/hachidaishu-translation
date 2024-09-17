# import logfire # TODO
import inspect
import logging
import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time}</green> <level>{level}</level> <blue>{module}:{function}:{line}</blue> <level>{message}</level>",
    # enqueue=True,
    # filter={
    #     "ssa": False,
    #     "byteflow": False,
    #     "interpreter": "ERROR",
    #     "typeinfer": "ERROR",
    # },
    level="INFO",
)
logger.add("run_{time}_log.json", level="DEBUG", enqueue=True, serialize=True)


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
