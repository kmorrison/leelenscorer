import cProfile
import datetime
from contextlib import contextmanager

@contextmanager
def profile(identifier=None):
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    if identifier is None:
        identifier = datetime.datetime.now().isoformat(sep='T')

    pr.dump_stats(f'{identifier}.prof')
