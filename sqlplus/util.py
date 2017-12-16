"""
Functions that are not specific to "Row" objects
"""
import random
import string
import concurrent.futures
import multiprocessing as mp
from itertools import chain, zip_longest

from datetime import datetime
from dateutil.relativedelta import relativedelta


def dateconv(date, infmt, outfmt):
    "date format conversion"
    return datetime.strftime(datetime.strptime(str(date), infmt), outfmt)


def pmap(fn, *args, max_workers=None):
    """ Parallel map
    """
    max_workers = min(max_workers if isinstance(max_workers, int) else 1,
                      mp.cpu_count())
    if max_workers == 1:
        yield from (fn(*a) for a in zip(*args))
    else:
        tempstr = random_string()
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for gs in grouper(zip(*args), max_workers, tempstr):
                gs = (x for x in gs if x != tempstr)
                yield from executor.map(fn, *zip(*gs))


# copied from 'itertools'
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def random_string(nchars=20):
    "Generates a random string of lengh 'n' with alphabets and digits. "
    chars = string.ascii_letters + string.digits
    return ''.join(random.SystemRandom().choice(chars)
                   for _ in range(nchars))


def peek_first(seq):
    """
    Note:
        peeked first item is pushed back to the sequence
    Args:
        seq (Iter[type])
    Returns:
        Tuple(type, Iter[type])
    """
    # never use tee, it'll eat up all of your memory
    seq1 = iter(seq)
    first_item = next(seq1)
    return first_item, chain([first_item], seq1)


# performance doesn't matter for this, most of the time
def listify(x):
    """
    Example:
        >>> listify('a, b, c')
        ['a', 'b', 'c']

        >>> listify(3)
        [3]

        >>> listify([1, 2])
        [1, 2]
    """
    try:
        return [x1.strip() for x1 in x.split(',')]
    except AttributeError:
        try:
            return list(iter(x))
        except TypeError:
            return [x]


# If the return value is True it is converted to 1 or 0 in sqlite3
# istext is unncessary for validity check
def isnum(*xs):
    "Tests if x is numeric"
    try:
        for x in xs:
            float(x)
        return True
    except:
        return False


def ymd(date, size, fmt):
    if isinstance(size, str):
        n, unit = size.split()
        if not unit.endswith('s'):
            unit = unit + 's'
        size = {unit: int(n)}
    d1 = datetime.strptime(str(date), fmt) + relativedelta(**size)
    d2 = d1.strftime(fmt)
    return int(d2) if isinstance(date, int) else d2


def star(val, pval, n=3):
    "put stars according to p-value"
    val = format(val, f'.{n}f')
    if pval < 0.01:
        return val + '***'
    elif pval < 0.05:
        return val + '**'
    elif pval < 0.1:
        return val + '*'
    else:
        return val



