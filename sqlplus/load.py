# TODO: This lib works but
# not cleaned up and not tested properly

import locale
import os
import csv

from itertools import groupby

import sqlplus
from sqlplus.core import Row
from sqlplus.util import grouper, listify

__all__ = ['fnguide']


if os.name == 'nt':
    locale.setlocale(locale.LC_ALL, 'english_USA')
elif os.name == 'posix':
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def fnguide(filename, cols=None):
    filename = os.path.join(sqlplus.core.WORKSPACE, filename)
    with open(filename, encoding='cp949') as f:
        # 8줄 버리고
        for _ in range(8):
            f.readline()
        reader = csv.reader(f)
        # checks if the number of 'cols' corresponds with the file
        line1 = next(reader)
        n, ids = extract_ids(line1)
        # 5줄 더 버리고
        for _ in range(5):
            next(reader)
        # cols given
        if cols:
            cols = listify(cols)
            assert len(cols) == n, f"Invalid cols given, {cols}, {n}"
            for line in csv.reader(f):
                for s, vs in zip(ids, grouper(line[1:], n)):
                    # 1,232,392 => interpret
                    vs1 = (convert_string(v) for v in vs)
                    r = Row()
                    r.date = line[0]
                    r.id = s
                    for c, v in zip(cols, vs1):
                        r[c] = v
                    yield r
        else:
            for line in csv.reader(f):
                for s, vs in zip(ids, grouper(line[1:], n)):
                    # date, id, col1, col2, ...
                    yield (line[0], s, *vs)


# doesnt have to be fast
def extract_ids(xs):
    result = []
    first_ids = None
    for _, ss in groupby(xs[1:], lambda x: x):
        ss = list(ss)
        if not first_ids:
            first_ids = ss
        # it could be just an empty string
        if ss[0].strip():
            result.append(ss[0].strip())
    return len(first_ids), result


def all_equal(lst):
    return (not lst) or len(set(lst)) == 1


def convert_string(x):
    # no comma
    if x.find(',') == -1:
        return x
    try:
        return locale.atoi(x)
    except ValueError:
        try:
            return locale.atof(x)
        except:
            return x

