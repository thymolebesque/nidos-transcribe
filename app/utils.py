import time
from contextlib import contextmanager
from typing import Iterator

@contextmanager
def stopwatch() -> Iterator[float]:
    start = time.time()
    yield start
    end = time.time()
    # no automatic print; callers compute end-start

def seconds() -> float:
    return time.time()
