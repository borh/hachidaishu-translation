import time

from hachidaishu_translation.log import logger


def retry(
    func,
    max_attempts: int = 3,
    delay: int = 1,
    exceptions=(Exception,),
    *args,
    **kwargs,
):
    """
    Retry a function up to `max_attempts` times with a delay between each attempt.

    Parameters:
    - func: The function to be retried.
    - max_attempts: Maximum number of retry attempts.
    - delay: Delay in seconds between retries.
    - exceptions: Tuple of exception types to catch for retries.
    - *args, **kwargs: Arguments and keyword arguments to pass to the function.

    Returns:
    - The return value of the function, if successful.

    Raises:
    - The last exception raised if all retry attempts fail.
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            return func(*args, **kwargs)
        except exceptions as e:
            attempt += 1
            logger.error(f"Attempt {attempt} failed: {e}")
            if attempt < max_attempts:
                time.sleep(delay)
            else:
                logger.error("Max attempts reached. Raising the last exception.")
                raise
