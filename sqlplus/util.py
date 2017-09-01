"""
Functions that are not specific to "Row" objects
"""
import random
import string
import fileinput
import pandas as pd

import concurrent.futures
import multiprocessing as mp
from itertools import chain, zip_longest

from datetime import datetime
from dateutil.relativedelta import relativedelta


def breakpoints(seq, percentiles):
    "Breakpoints from percentages"
    bs = pd.Series(seq).describe(percentiles)
    return [bs[str(round(p * 100)) + '%'] for p in percentiles]


def read_date(date, infmt, outfmt="%Y%m%d"):
    """converts date formt

    read_date('31Mar2013', '%d%b%Y')
    """
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


def prepend_header(filename, header=None, drop=0):
    """Drop n lines and prepend header

    Args:
        filename (str)
        header (str)
        drop (int)
    """
    for no, line in enumerate(fileinput.input(filename, inplace=True)):
        # it's meaningless to set drop to -1, -2, ...
        if no == 0 and drop == 0:
            if header:
                print(header)
            print(line, end='')
        # replace
        elif no + 1 == drop:
            if header:
                print(header)
        elif no >= drop:
            print(line, end='')
        else:
            # no + 1 < drop
            continue


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
def isnum(*xs):
    "Tests if x is numeric"
    try:
        for x in xs:
            float(x)
        return True
    except:
        return False


def ymd(step, fmt='%Y%m%d'):
    def add_datetime(n, unit):
        def fn(date):
            d1 = datetime.strptime(str(date), fmt) + relativedelta(**{unit: n})
            d2 = d1.strftime(fmt)
            return int(d2) if isinstance(date, int) else d2
        return fn

    if isinstance(step, str):
        try:
            n, unit = step.split()
            n, unit = int(n), unit.lower()
            if not unit.endswith('s'):
                unit += 's'
        except:
            raise ValueError(f"Invalid format {step}")
        return add_datetime(n, unit)

    elif isinstance(step, int):
        return lambda date: date + step
    else:
        raise ValueError(f"Invalid format {step}")


# Not so fast but general and quits if not all equal
def same(iterator):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def parse_model(model):
    "y ~ x1 + x2 => ['y', 'x1', 'x2']"
    left, right = model.split('~')
    return [left.strip()] + [x.strip() for x in right.split('+')]


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

