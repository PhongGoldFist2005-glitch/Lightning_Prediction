import logging

def create_logger(log_file):
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    # tránh add handler nhiều lần
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger