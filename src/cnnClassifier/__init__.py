import os
import sys
import logging

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)


# Configure the logging system
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=logging_str,  # Set the log message format
    handlers=[
        logging.FileHandler(log_filepath),  # Log messages to a file
        logging.StreamHandler(sys.stdout)  # Print log messages to the console
    ]
)

# Create a logger with the name "cnnClassifierLogger"
logger = logging.getLogger("cnnClassifierLogger")

# Loggers are used to emit log messages, and by naming them, you can control 
# the granularity of logging in your application. Multiple loggers can exist, 
# each emitting messages at different levels and to different destinations.

