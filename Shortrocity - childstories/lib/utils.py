# wait for n random minutes
import random
import time
import logging

def wait_for_n_random_minutes(n: int):
    wait_time = random.randint(1, 1 + 4 * 60)
    logging.info(f"Waiting for {wait_time/60} minutes.")
    time.sleep(wait_time)