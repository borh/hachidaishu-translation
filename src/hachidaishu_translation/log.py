import logging
from datetime import datetime

# Set the root logger to a higher level to prevent it from handling lower-level logs
logging.getLogger().setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Remove all handlers associated with the root logger object
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Create a console handler and set the level to info
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a file handler and set the level to info
log_filename = datetime.now().strftime("run_%Y%m%d_%H%M%S.log")
file_handler = logging.FileHandler(log_filename, mode="w")
file_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)
