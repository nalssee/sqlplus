"""
Functions that are not specific to "Row" objects
"""
import random
import string
from itertools import chain, zip_longest

from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import wraps


def perr(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            yield from fn(*args, **kwargs)
        except Exception as e:
            print(e)
    return wrapper


def dmath(date, infmt, outfmt=None, **size):
    """Date arithmetic
    Returns int if input(date) is int else str
    """
    outfmt = outfmt or infmt
    if not size:
        # Just convert the format
        return datetime.strftime(datetime.strptime(str(date), infmt), outfmt)
    d1 = datetime.strptime(str(date), infmt) + relativedelta(**size)
    d2 = d1.strftime(outfmt)
    return int(d2) if isinstance(date, int) else d2


# If the return value is True it is converted to 1 or 0 in sqlite3
# istext is unncessary for validity check
def isnum(*xs):
    "Tests if x is numeric"
    try:
        for x in xs:
            float(x)
        return True
    except (ValueError, TypeError):
        return False


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def _random_string(nchars=20):
    "Generates a random string of lengh 'n' with alphabets and digits. "
    chars = string.ascii_letters + string.digits
    return ''.join(random.SystemRandom().choice(chars)
                   for _ in range(nchars))


def _peek_first(seq):
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
def _listify(x):
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
