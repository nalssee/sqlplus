import locale
import csv
import xlrd
import openpyxl
import numpy as np
from itertools import accumulate, groupby, chain, islice
import pandas as pd


def lag(cols, datecol, ns, add1fn=None, max_missings=10_000):
    """ create columns for lags and leads, ex) col_1, col_2n
    """
    cols = listify(cols)
    ns = listify(ns)

    def fn(rs):
        suffix = "_"
        rs.sort(key=lambda r: r[datecol])
        if add1fn:
            rs1 = [rs[0]]
            for r1, r2 in zip(rs, rs[1:]):
                d1, d2 = r1[datecol], r2[datecol]
                if d1 == d2:
                    raise ValueError("Duplicates for lead lags", r1, r2)
                if add1fn(d1) == d2:
                    rs1.append(r2)
                else:
                    cnt = 0
                    d1 = add1fn(d1)
                    while d1 != d2:
                        if cnt >= max_missings:
                            raise ValueError("Too many missings", r1, r2)
                        r0 = _empty(r1)
                        r0[datecol] = d1
                        rs1.append(r0)
                        d1 = add1fn(d1)
                        cnt += 1
                    rs1.append(r2)
        else:
            rs1 = rs

        r0 = rs1[0]
        rss = [rs1]
        for n in ns:
            if n >= 0:
                rss.append([_empty(r0)] * n + rs1)
            else:
                rss.append(rs1[(-n):] + [_empty(r0)] * (-n))

        result = []
        for xs in zip(*rss):
            x0 = xs[0]
            for n, x in zip(ns, xs[1:]):
                for c in cols:
                    strn = str(n) if n >= 0 else str(-n) + 'n'
                    x0[c + suffix + strn] = x[c]
            result.append(x0)
        return result
    return fn


# You should be very careful if you want to update this generator
def step(grouped_seqs, stop_short=False):
    """ Generates tuples of lists of rows for every matching keys
    """
    Empty = object()
    NoMore = object()
    EmptyVal = []

    keys = [Empty] * len(grouped_seqs)
    vals = [EmptyVal] * len(grouped_seqs)

    def update(i, gs):
        try:
            k, rs = next(gs)
            keys[i] = k or Empty
            vals[i] = list(rs)
        except StopIteration:
            keys[i] = NoMore
            vals[i] = EmptyVal

    for i, gs in enumerate(grouped_seqs):
        update(i, gs)

    # Dont worry about function call overhead for thunks.
    # Function call overhead is mostly due to arg type checking
    if stop_short:
        def continuation_condition(): return all(k is not NoMore for k in keys)
    else:
        def continuation_condition(): return not all(k is NoMore for k in keys)

    while continuation_condition():
        try:
            minkey = min(k for k in keys if k is not NoMore)
        except TypeError as e:
            # remove empty ones
            if Empty in keys:
                minkey = Empty
            else:
                # 'abc' < 3
                raise e

        result1 = []
        for i, (truth, k, v, g) in\
                enumerate(zip((k == minkey for k in keys), keys,
                              vals, grouped_seqs)):
            if truth:
                update(i, g)
                result1.append(v)
            else:
                result1.append(EmptyVal)
        yield result1


def where(pred, fn=None):
    """ Filter with pred before you apply fn to a list of rows.
        if fn is not given, simply filter with pred
    """
    if fn:
        def func(rs):
            return fn([r for r in rs if pred(r)])
        return func
    else:
        return lambda r: (r if pred(r) else None)


def add(**kwargs):
    # vs is a list of functions
    ks, vs = list(kwargs), list(kwargs.values())

    if len(ks) == 1:
        k1, v1 = ks[0], vs[0]

    if len(ks) == 2:
        k1, v1 = ks[0], vs[0]
        k2, v2 = ks[1], vs[1]

    def _f1(x):
        x[k1] = v1(x)
        return x

    def _f2(x):
        x[k1], x[k2] = v1(x), v2(x)
        return x

    def _fn(x):
        for k, v in kwargs.items():
            x[k] = v(x)
        return x

    if len(ks) == 1:
        return _f1
    elif len(ks) == 2:
        return _f2
    else:
        return _fn


def head(n):
    def _f(rs):
        yield from islice(rs, 0, n)
    return _f


def read_date(date):
    """ Read date string and returns a python datetime object
    """
    return pd.to_datetime(date)


def add_date(date, n):
    """n means days for "%Y-%m-%d", months for "%Y-%m". Supports only 2 fmt.
    """
    # "1903-09"
    if len(date) == 7:
        y, m = date.split("-")
        n1 = int(y) * 12 + int(m) + n
        y1, m1 = n1 // 12, n1 % 12
        if m1 == 0:
            y1 -= 1
            m1 = 12
        return str(y1) + '-' + str(m1).zfill(2)
    # "1903-09-29"
    elif len(date) == 10:
        d = pd.to_datetime(date)
        d1 = d + pd.Timedelta(days=n)
        return d1.strftime("%Y-%m-%d")
    else:
        raise ValueError("Unsupported date format", date)


def getX(rs, cols, constant=False):
    """ Returns a list of lists (matrix X) for linear regression,
    if constant is True, X with a constant(1) column.

    :param rs: list of rows
    :param cols: comma separated str for columns to be selected
    :param constant: True or False

    :returns: list of lists (matrix X)
    """
    def getvals_fn(cols, const):
        if const:
            return lambda r: [1] + [r[c] for c in cols]
        else:
            return lambda r: [r[c] for c in cols]

    getvals = getvals_fn(listify(cols), constant)
    return [getvals(r) for r in rs]


def gety(rs, col):
    """ Returns a list of numbers (matrix X) for linear regression

    :param rs: a list of
    """
    return [r[col] for r in rs]


def set_default(rs, cols, val=''):
    cols = listify(cols)
    for r in rs:
        for c in cols:
            r[c] = val


def chunk(rs, n, column=None):
    """Returns a list of rows in chunks

    :param rs: a list of rows
    :param n:
        - int => returns 3 rows about the same size
        - list of ints, [0.3, 0.4, 0.3] => returns 3 rows of 30%, 40$, 30%
        - list of nums, [100, 500, 100] => returns 4 rows with break points\
        100, 500, 1000, but you must pass the column name\
        for the break points like

        chunk(rs, [100, 500, 100], 'col')
    :param column: column name for break points
    :returns: a list of rows
    """
    size = len(rs)
    if isinstance(n, int):
        start = 0
        result = []
        for i in range(1, n + 1):
            end = int((size * i) / n)
            # must yield anyway
            result.append(rs[start:end])
            start = end
        return result
    # n is a list of percentiles
    elif not column:
        # then it is a list of percentiles for each chunk
        assert sum(n) <= 1, "Sum of percentils for chunks must be <= 1.0"
        ns = [int(x * size) for x in accumulate(n)]
        result = []
        for a, b in zip([0] + ns, ns):
            result.append(rs[a:b])
        return result
    # n is a list of break points
    else:
        rs.sort(key=lambda r: r[column])
        start, end = 0, 0
        result = []
        for bp in n:
            while (rs[end][column] < bp) and end < size:
                end += 1
            result.append(rs[start:end])
            start = end
        result.append(rs[end:])
        return result


def isnum(*xs):
    """ Tests if all of xs are numeric

    :param xs: vals of whatever

    :returns: True or False
    """
    try:
        for x in xs:
            float(x)
        return True
    except (ValueError, TypeError):
        return False


def allnum(rs, cols):
    """Returns a list of rows that all of columns are numeric.

    :param rs: [r1, r2, ...]
    :type rs: list of rows
    :param cols: 'col1, col2'
    :type cols: comma separated string

    :returns: list of rows
    """
    cols = listify(cols)
    return [r for r in rs if all(isnum(r[c]) for c in cols)]


def stars(pval):
    """ Returns a string of stars based on pvalue
    """
    if pval <= 0.01:
        return "***"
    elif pval <= 0.05:
        return "**"
    elif pval <= 0.10:
        return "*"
    return ""


def listify(x):
    """
    Attempt to turn it into a list if possible
    if not return as is
    """
    try:
        return [x1.strip() for x1 in x.split(',')]
    except AttributeError:
        try:
            return list(iter(x))
        except TypeError:
            # x not [x]
            return x


def readxl(fname, sheets=None, encoding='utf-8', delimiter=None,
           quotechar='"', newline='\n', by_sheet=False):
    """ Reads excel like files and yields a list of values
    """
    # locale is set in fb.run()
    def conv(x):
        try:
            return locale.atoi(x)
        except ValueError:
            try:
                return locale.atof(x)
            except Exception:
                return x
    # csv, tsv, ssv ...
    if not (fname.endswith('.xls') or fname.endswith('.xlsx')):
        # default delimiter is ","
        delimiter = delimiter or ("\t" if fname.lower().endswith('.tsv')
                                  else ",")
        with open(fname, encoding=encoding, newline=newline) as fin:
            for rs in csv.reader((x.replace('\0', '') for x in fin),
                                 delimiter=delimiter, quotechar=quotechar):
                yield [conv(x) for x in rs]

    elif fname.endswith('xls'):
        # openpyxl does not work with the old ".xls" format
        def getval(cell):
            if cell.ctype == xlrd.XL_CELL_EMPTY or\
               cell.ctype == xlrd.XL_CELL_TEXT:
                return cell.value.strip()
            elif cell.value.is_integer():
                return int(cell.value)
            else:
                return cell.value

        workbook = xlrd.open_workbook(fname)
        sheets = listify(sheets) if sheets else [workbook.sheets()[0].name]
        if sheets == ['*']:
            sheets = [sheet.name for sheet in workbook.sheets()]
        if by_sheet:
            for sheet in sheets:
                lines = ([getval(c) for c in r]
                         for r in workbook.sheet_by_name(sheet).get_rows())
                yield (sheet, lines)
        else:
            for sheet in sheets:
                for row in workbook.sheet_by_name(sheet).get_rows():
                    yield [getval(c) for c in row]
    else:
        # openpyxl for xlsx
        def getval(cell):
            val = cell.value
            return val.strip() if isinstance(val, str) else val

        workbook = openpyxl.load_workbook(fname, read_only=True,
                                          data_only=True, keep_links=False)
        sheets = listify(sheets) if sheets else [workbook.sheetnames[0]]
        if sheets == ['*']:
            sheets = workbook.sheetnames
        if by_sheet:
            for sheet in sheets:
                lines = ([getval(c) for c in r] for r in workbook[sheet].rows)
                yield (sheet, lines)
        else:
            for sheet in sheets:
                for row in workbook[sheet].rows:
                    yield [getval(c) for c in row]


# implicit ordering
def group(rs, key):
    keyfn = _build_keyfn(key)
    rs.sort(key=keyfn)
    return [list(rs1) for _, rs1 in groupby(rs, keyfn)]


def overlap(rs, size, step=1, key=None):
    if key:
        xs = group(rs, key)
        return [list(chain(*xs[i:i + size])) for i in range(0, len(xs), step)]
    else:
        return [rs[i:i + size] for i in range(0, len(rs), step)]


def avg(rs, col, wcol=None, ndigits=None):
    """ Returns the average value of the given column.

    :param rs: a list of rows
    :param col: str (column name)
    :param wcol: str (column for weighted average)
    :param ndigits: int (number of digits to the right of the decimal point)

    :returns: a float
    """
    if wcol:
        xs = [r for r in rs if isnum(r[col], r[wcol])]
        val = np.average([x[col] for x in xs], weights=[x[wcol] for x in xs])\
            if xs else ''
    else:
        xs = [r for r in rs if isnum(r[col])]
        val = np.average([x[col] for x in xs]) if xs else ''
    return round(val, ndigits) if ndigits and xs else val


def _empty(r):
    return {c: '' for c in r}


def _build_keyfn(key):
    " if key is a string return a key function "
    # if the key is already a function, just return it
    if hasattr(key, '__call__'):
        return key
    colnames = listify(key)
    # special case
    if colnames == ['*']:
        return lambda r: 1

    if len(colnames) == 1:
        col = colnames[0]
        return lambda r: r[col]
    return lambda r: [r[colname] for colname in colnames]
