"""
Functions that are not specific to "Row" objects
"""
import random
import string
from itertools import chain

from datetime import datetime
from dateutil.relativedelta import relativedelta


def dconv(date, infmt, outfmt):
    """Converts date format

    Args:
        |  date(int or str): 199912 or '1999DEC'
        |  infmt(str): input format
        |  outfmt(str): output format

    Date format examples:
        |  %Y, %m, %d, %b ...
        |  https://docs.python.org/3/library/datetime.html

    Returns str
    """
    return datetime.strftime(datetime.strptime(str(date), infmt), outfmt)


def dmath(date, size, fmt):
    """Date arithmetic

    Args:
        |  date(int or str): 19991231 or "1999-12-31'
        |  size(str): "3 months"
        |  fmt(str): date format

    Returns int if input(date) is int else str
    """
    if isinstance(size, str):
        n, unit = size.split()
        if not unit.endswith('s'):
            unit = unit + 's'
        size = {unit: int(n)}
    d1 = datetime.strptime(str(date), fmt) + relativedelta(**size)
    d2 = d1.strftime(fmt)
    return int(d2) if isinstance(date, int) else d2


def isconsec(xs, size, fmt):
    """Tests if xs is consecutive calendrically, increasing order.
    """
    for x1, x2 in zip(xs, xs[1:]):
        if dmath(x1, size, fmt) != x2:
            return False
    return True


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
