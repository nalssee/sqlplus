"""
sqlite3 based utils for statistical analysis

reeling off rows from db(sqlite3) and saving them back to db
"""
import os
import sys
import csv
import re
import sqlite3
import io
import copy
import warnings
import inspect
import platform
import operator
import tempfile

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statistics as st

from collections import Counter, OrderedDict
from contextlib import contextmanager
from itertools import groupby, islice, chain, product, tee, zip_longest, repeat, accumulate
from pandas.tools.plotting import scatter_matrix


from sas7bdat import SAS7BDAT

from .util import isnum, listify, camel2snake, peek_first, \
    parse_model, random_string, pmap


__all__ = ['dbopen', 'Row', 'Rows', 'Box']


WORKSPACE = ''


@contextmanager
def dbopen(dbfile, cache_size=100000, temp_store=2):
    # temp_store might be deprecated
    "Connects to SQL database(sqlite)"
    splus = SQLPlus(dbfile, cache_size, temp_store)
    try:
        yield splus
    finally:
        # should I close the cursor?
        splus._cursor.close()
        splus.conn.commit()
        splus.conn.close()


# aggreate function builder
class AggBuilder:
    def __init__(self):
        self.rows = []

    def step(self, *args):
        self.rows.append(args)

    def finalize(self):
        return self.rows


# Don't try to be smart, unless you really know well
class Row:
    "mutable version of sqlite3.row"
    # works for python 3.6 and higher
    def __init__(self, **kwargs):
        super().__setattr__('_ordered_dict', OrderedDict(**kwargs))

    def copy(self):
        r = Row()
        for c, v in zip(self.columns, self.values):
            r[c] = v
        return r

    @property
    def columns(self):
        return list(self._ordered_dict.keys())

    @property
    def values(self):
        return list(self._ordered_dict.values())

    def __getattr__(self, name):
        return self._ordered_dict[name]

    def __setattr__(self, name, value):
        self._ordered_dict[name] = value

    def __delattr__(self, name):
        del self._ordered_dict[name]

    def __getitem__(self, name):
        return self._ordered_dict[name]

    def __setitem__(self, name, value):
        self._ordered_dict[name] = value

    def __delitem__(self, name):
        del self._ordered_dict[name]

    def __repr__(self):
        content = ', '.join(c + '=' + repr(v) for c, v in
                             zip(self.columns, self.values))
        return 'Row(' + content + ')'

    # for pickling, very important
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    # todo
    # hasattr doesn't for properly
    # you can't make it work by changing getters and setters
    # to an ordinary way. but it is slower


class Rows:
    """
    a shallow wrapper of a list of row instances """
    # don't try to define __getattr__, __setattr__
    # list objects has a lot of useful attributes that can't be overwritten
    # not the same situation as 'row' class

    # inheriting list can be problemetic
    # when you want to use this as a superclass
    # see 'where' method, you must return 'self' but it's not efficient
    # (at least afaik) if you inherit list

    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    # __getitem__ enables you to iterate 'Rows'
    def __getitem__(self, cols):
        "cols: integer or list of strings or comma separated string"
        if isinstance(cols, int):
            return self.rows[cols]
        elif isinstance(cols, slice):
            # shallow copy for non-destructive slicing
            self = self.copy()
            self.rows = self.rows[cols]
            return self

        cols = listify(cols)
        if len(cols) == 1:
            col = cols[0]
            return [r[col] for r in self.rows]
        else:
            return [[r[c] for c in cols] for r in self.rows]

    def __setitem__(self, cols, vals):
        """vals can be just a list or a list of lists,
        demensions must match
        """
        if isinstance(cols, int) or isinstance(cols, slice):
            self.rows[cols] = vals
            return

        cols = listify(cols)
        ncols = len(cols)

        if not isinstance(vals, list):
            if ncols == 1:
                col = cols[0]
                for r in self.rows:
                    r[col] = vals
            else:
                for r in self.rows:
                    for c in cols:
                        r[c] = vals

        elif not isinstance(vals[0], list):
            if ncols != len(vals):
                raise ValueError('Number of values to assign inappropriate')
            for r in self.rows:
                for c, v in zip(cols, vals):
                    r[c] = v

        else:
            # validity check,
            if len(self.rows) != len(vals):
                raise ValueError('Number of values to assign inappropriate')

            # vals must be rectangular!
            if ncols > 1:
                for vs in vals:
                    if len(vs) != ncols:
                        raise ValueError('Invalid values to assign', vs)

            if ncols == 1:
                col = cols[0]
                for r, v in zip(self.rows, vals):
                    r[col] = v
            else:
                for r, vs in zip(self.rows, vals):
                    for c, v in zip(cols, vs):
                        r[c] = v

    def __delitem__(self, cols):
        if isinstance(cols, int) or isinstance(cols, slice):
            del self.rows[cols]
            return

        cols = listify(cols)
        ncols = len(cols)

        if ncols == 1:
            col = cols[0]
            for r in self.rows:
                del r[col]
        else:
            for r in self.rows:
                for c in cols:
                    del r[c]

    def __add__(self, other):
        self.rows = self.rows + other.rows
        return self

    # zipping like left join, based on column
    def lzip(self, col, *rss):
        """self and rss are all ordered(ascending) and
        none of them contains dups
        Parameters:
            rss: a list of a sequence of instances of 'Row',
                a sequence can be either a list of iterator
        """
        # This should be fast, so the verification steps are ignored
        rss = [iter(rs) for rs in rss]
        def gen(rs0, rs1):
            doneflag = False
            try:
                r1 = next(rs1)
            except:
                doneflag = True

            for r0 in rs0:
                if doneflag:
                    yield None
                    continue
                v0, v1 = r0[col], r1[col]
                if v0 < v1:
                    yield None
                    continue
                elif v0 == v1:
                    yield r1
                    try:
                        r1 = next(rs1)
                    except:
                        doneflag = True
                else:
                    while v0 > v1:
                        try:
                            # throw away
                            r1 = next(rs1)
                            v1 = r1[col]
                        except:
                            doneflag = True
                            break
                        # nothing to yield
                    if v0 == v1:
                        yield r1
                    # passed over
                    else:
                        yield None

        rs0s = tee(iter(self), len(rss) + 1)
        seqs = (gen(rs0, rs1) for rs0, rs1 in zip(rs0s[1:], rss))
        yield from zip(rs0s[0], *seqs)


    # dependent portfolio
    def _dpn(self, cps):
        prev_cols = []
        for c, ps in cps.items():
            pncol = 'pn_' + c
            for rs in self.group(prev_cols):
                for i, rs1 in enumerate(rs.order(c)._chunks(ps, c), 1):
                    rs1[pncol] = i
            prev_cols.append(pncol)
        return self

    # indepnedent
    def _ipn(self, cps):
        for c, ps in cps.items():
            pncol = 'pn_' + c
            for i, rs in enumerate(self.order(c)._chunks(ps, c), 1):
                rs[pncol] = i
        return self


    def portfolios(self, cps, dcol, icol, dep=False):
        "Number portfolios based on the first date"
        # self can't be empty

        if isinstance(cps, dict):

            cols = list(cps)
            pncols = ['pn_' + c for c in cols]
            self[pncols] = ''
            self.order(dcol)
            rs = next(self.group(dcol, order=False))

            if dep:
                rs._dpn(cps)
            else:
                rs._ipn(cps)
            # python sort preserves order
            for rs1 in self.group(icol):
                for pncol in pncols:
                    rs1[pncol] = rs1[0][pncol]
            return self
        # cps is an instance of Rows
        pncols = [c for c in cps[0].columns if c.startswith('pn_') \
                  and not c.endswith('_max') and not c.endswith('_min')]
        cols = [c[3:] for c in pncols]
        self[pncols] = ''
        self.order(dcol)
        rs = next(self.group(dcol, order=False))

        for r in cps.where({dcol: rs[0][dcol]}):
            rs1 = rs.where({c: [r[c + '_min'], r[c + '_max']] for c in pncols})
            for c in pncols:
                rs1[c] = r[c]
        return self


    def breaks(self, cps, dcol, dep=False):
        def accum(xs):
            return [xs[:i] for i in range(1, len(xs) + 1)]

        cols = list(cps)
        pncols = ['pn_' + c for c in cols]
        sizes = [len(v) for v in cps.values()]
        date = self[0][dcol]
        # dependent breaks for one date
        def _dbrks():
            self._dpn(cps)
            d = {}
            def ab(pns, rs, col, size):
                *pn0, n = pns
                if n == 1:
                    # already well ordered. so no need to use
                    return (float('-inf'), rs[-1][col])
                elif n < size:
                    return (d[(*pn0, n - 1)], rs[-1][col])
                else:
                    return (rs[-1][col], float('+inf'))

            for pncols1 in accum(pncols):
                for rs in self.group(pncols1, order=False):
                    r0 = rs[0]
                    pns = tuple(r0[c] for c in pncols1)
                    col = cols[len(pns) - 1]
                    size = sizes[len(pns) - 1]
                    d[pns] = d.get(pns[:-1], []) + list(ab(pns, rs, col, size))

            for pns in product(*(range(1, s + 1) for s in sizes)):
                try:
                    yield pns, d[pns]
                except:
                    raise ValueError(f'Not enough obs at {date}')

        def _ibrks():
            self._ipn(cps)
            intvals = []
            for col, pncol in zip(cols, pncols):
                self.order(col)
                bps1 = [rs[-1][col] for rs in self.group(pncol)]
                intvals1 = [(a, b) for a, b in zip([float('-inf')] + bps1, bps1[:-1] + [float('+inf')])]
                intvals.append(intvals1)

            for pns, intval in zip_longest(product(*(range(1, s + 1) for s in sizes)),
                                           product(*intvals)):

                assert intval, f'Not enough obs at {date}'
                yield pns, intval
        result = []
        for (pns, intval) in (_dbrks() if dep else _ibrks()):
            r = Row()
            r[dcol] = date
            for pncol, pn1, (a, b) in zip(pncols, pns, intval):
                r[pncol] = pn1
                r[pncol + '_min'] = a
                r[pncol + '_max'] = b
            result.append(r)
        return self._newrows(result)



    # self should be ordered first, You may or may not need col
    def _chunks(self, ps, col):
        # ps: int or [3, 4, 3]
        #         or [[0, 0], 4, 5]
        # probably you don't have to pass col but I'm being cautious
        def bpred(col, op1, a1, op2=None, a2=None):
            # build predicate
            def fn1(r):
                return op1(r[col], a1)
            def fn2(r):
                return op1(r[col], a1) and op2(r[col], a2)
            return fn2 if (op2 and a2) else fn1

        def cks(rs, ps):
            # chunks

            n = len(rs)
            tot = sum(ps)
            ps1 = list(p * n / tot for p in accumulate(ps))
            # just to be safe
            ps2 = ps1[:-1] + [n]
            for a, b in zip([0] + ps2, ps2):
                yield self[a:b]

        if isinstance(ps, int):
            # take care of cutoffs, just in case
            yield from cks(self, [1 for _ in range(ps)])

        # gotta be a list then
        else:
            breaks = []
            for i, p in enumerate(ps):
                if isinstance(p, list):
                    # must be a list of len 2
                    a, b = p
                    breaks.append((a, i, True))
                    breaks.append((b, i, True))
                elif isinstance(p, tuple):
                    a, b = p
                    breaks.append((a, i, False))
                    breaks.append((b, i, False))
            if breaks == []:
                yield from cks(self, ps)
            else:
                a, i, inclusive = breaks[0]
                if i != 0:
                    op = operator.lt if inclusive else operator.le
                    pred = bpred(col, op, a)
                    yield from cks(self.where(pred, True), ps[:i])
                for (a1, i1, inc1), (a2, i2, inc2) in zip(breaks, breaks[1:]):
                    if i1 == i2:
                        op1 = operator.ge if inc1 else operator.gt
                        op2 = operator.le if inc1 else operator.lt
                        pred = bpred(col, op1, a1, op2, a2)
                        yield self.where(pred, True)
                    elif i1 + 1 == i2:
                        continue
                    else:
                        op1 = operator.gt if inc1 else operator.ge
                        op2 = operator.lt if inc2 else operator.le
                        pred = bpred(col, op1, a1, op2, a2)
                        yield from cks(self.where(pred, True), ps[i1 + 1:i2])

                a, i, inclusive = breaks[-1]
                # if not the last one
                if i != len(ps) - 1:
                    op = operator.gt if inclusive else operator.ge
                    pred = bpred(col, op, a)
                    yield from cks(self.where(pred, True), ps[i+1:])


    def roll(self, period, jump, dcol, nextfn):
        # dcol: date column
        # inc: increment function, which might be deprecated
        "group rows over time, allowing overlaps"
        if len(self.rows) == 0:
            yield self._newrows([])
        else:
            # Python sort is stable(order preserving)
            self.order(dcol)
            for ls in _roll(self.rows, period, jump, _build_keyfn(dcol), nextfn):
                yield self._newrows(ls)

    # destructive!!!
    def order(self, key, reverse=False):
        self.rows.sort(key=_build_keyfn(key), reverse=reverse)
        return self

    def copy(self):
        "shallow copy"
        # I'm considering the inheritance
        return copy.copy(self)

    def _newrows(self, rs):
        # copying rows and build Rows object
        # Am I worring too much?, this is for inheritance
        self.rows, temp = [], self.rows
        other = self.copy()
        other.rows, self.rows = list(rs), temp
        return other


    def where(self, predx, ordered_rows=False):
        """
        rs.where(col1=3, col2=[1991, 2003], col3={'A003', 'A212'})
        """
        if isinstance(predx, dict):
            preds = []
            for k, v in predx.items():
                if isinstance(v, list) or isinstance(v, tuple):
                    a, b = v
                    op1 = operator.ge if isinstance(v, list) else operator.gt
                    if a and b:
                        preds.append((lambda k, a, b, op1: lambda r: op1(r[k], a) \
                                      and op1(b, r[k]))(k, a, b, op1))
                    elif a:
                        preds.append((lambda k, a, op1: lambda r: op1(r[k], a))(k, a, op1))
                    else:
                        preds.append((lambda k, b, op1: lambda r: op1(b, r[k]))(k, b, op1))
                elif isinstance(v, set):
                    preds.append((lambda k, v: lambda r: r[k] in v)(k, v))
                else:
                    preds.append((lambda k, v: lambda r: r[k] == v)(k, v))
            def pred(r):
                try:
                    return all(p(r) for p in preds)
                except:
                    return False
            return self._filter(pred, ordered_rows)
        return self._filter(predx, ordered_rows)


    def _filter(self, pred, ordered_rows=False):
        if ordered_rows:
            try:
                n = len(self)
                a = next(i for i in range(n) if pred(self[i]))
                try:
                    b = next(i for i in range(a + 1, n) if not pred(self[i]))
                    return self._newrows(self[a:b])
                except:
                    return self._newrows(self[a:])
            except:
                return self._newrows([])
        else:
            return self._newrows(r for r in self if pred(r))

    def summary(self, cols=None, percentile=None):
        def fill(xs, cols):
            d = {}
            for a, b in zip(xs.index, xs):
                d[a] = b

            result = []
            for c in cols:
                if c not in d:
                    result.append(float('nan'))
                else:
                    result.append(d[c])
            return result

        percentile = percentile or [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        df = self.df(cols)
        desc = df.describe(percentile, include='all')
        desc.loc['skewness'] = fill(df.skew(), desc.columns)
        desc.loc['kurtosis'] = fill(df.kurtosis(), desc.columns)
        return Rows(_df_reel(desc, True))

    def corr(self, cols=None):
        cols = cols or self[0].columns
        df = self.df(cols)
        corr1 = df.corr()
        corr2 = df.corr('spearman')
        columns = list(corr1.columns.values)
        c0 = _gen_valid_column_names(['p_s'] + columns)[0]

        lcorr1 = corr1.values.tolist()
        lcorr2 = corr2.values.tolist()
        for i in range(len(columns)):
            for j in range(i):
                lcorr2[i][j] = lcorr1[i][j]
        for i in range(len(columns)):
            lcorr2[i][i] = ''
        result = []
        for c, ls in zip(columns, lcorr2):
            r = Row()
            r[c0] = c
            for c, x in zip(columns, ls):
                r[c] = x
            result.append(r)
        return self._newrows(result)


    #  allowing *cols is because this is going to be used in SQL
    def isnum(self, *cols):
        "another simplified filtering, numbers only"
        cols = listify(','.join(cols))
        return self._newrows(r for r in self if isnum(r[c] for c in cols))

    # not used often
    def istext(self, *cols):
        "another simplified filtering, texts(string) only"
        cols = listify(','.join(cols))
        return self._newrows(r for r in self if all(isinstance(r[c], str) for c in cols))

    def wavg(self, col, wcol=None):
        if wcol:
            rs = self.isnum(col, wcol)
            total = sum(r[wcol] for r in rs)
            return sum(r[col] * r[wcol] / total for r in rs)
        else:
            return st.mean(r[col] for r in self if isnum(r[col]))


    def ols(self, model):
        # Remove the following later
        warnings.filterwarnings("ignore")
        y, *xs = parse_model(model)
        res = sm.OLS(self[y], sm.add_constant(self[xs])).fit()
        c0 = _gen_valid_column_names(['var'] + xs)[0]
        result = []
        def addrow(var):
            r = Row()
            r[c0] = var
            for a, b in zip(xs, getattr(res, var)):
                r[a] = b
            result.append(r)
        addrow('params')
        addrow('tvalues')
        addrow('pvalues')
        return self._newrows(result), res


    def plot(self, cols=None):
        cols = listify(cols) if cols else self[0].columns
        scatter_matrix(self.isnum(*cols).df(cols))
        plt.show()

    def truncate(self, col, limit=0.01):
        "Truncate extreme values, defalut 1 percent on both sides"
        xs = self[col]
        lower = np.percentile(xs, limit * 100)
        higher = np.percentile(xs, (1 - limit) * 100)
        return self.where(lambda r: r[col] >= lower and r[col] <= higher)

    def winsorize(self, col, limit=0.01):
        xs = self[col]
        lower = np.percentile(xs, limit * 100)
        higher = np.percentile(xs, (1 - limit) * 100)
        for r in self.rows:
            if r[col] > higher:
                r[col] = higher
            elif r[col] < lower:
                r[col] = lower
        return self

    def group(self, key, order=True):
        keyfn = _build_keyfn(key)
        if order:
            self.order(keyfn)
        for _, rs in groupby(self, keyfn):
            yield self._newrows(rs)

    def show(self, n=None, cols=None, file=None, excel=False):
        if not file:
            n = n if n else 10
            if self == []:
                print(self.rows)
            else:
                _show(self.rows, n, cols)
        else:
            file = file if isinstance(file, str) else sys.stdout
            rows = self.rows[:n] if (isinstance(n, int) and n > 0) else self.rows
            _csv(rows, file, cols)
            _open_excel(rows, file)



    # Use this when you need to see what's inside
    # for example, when you want to see the distribution of data.
    def df(self, cols=None):
        if cols:
            cols = listify(cols)
            return pd.DataFrame([[r[col] for col in cols] for r in self.rows],
                                columns=cols)
        else:
            cols = self.rows[0].columns
            seq = _safe_values(self.rows, cols)
            return pd.DataFrame(list(seq), columns=cols)




class Box:
    """We need something very simple and flexible for displaying
    list of lists
    """
    def __init__(self, lines):
        self.lines = lines

    def show(self, file=sys.stdout, excel=False):
        _csv(self.lines, file, None)
        _open_excel(file, excel)
    # need some tools to build a Box easily


class SQLPlus:
    def __init__(self, dbfile, cache_size, temp_store):
        """
        Args:
            dbfile (str): db filename or ':memory:'
        """
        global WORKSPACE


        # set workspace if it's not there
        if not WORKSPACE:
            if os.path.isabs(dbfile):
                WORKSPACE = os.path.dirname(dbfile)
            elif os.path.dirname(dbfile):
                WORKSPACE = os.path.join(os.getcwd(), os.path.dirname())
            else:
                # default workspace
                WORKSPACE = os.path.join(os.getcwd(), 'workspace')

        if not os.path.exists(WORKSPACE):
            os.makedirs(WORKSPACE)

        if dbfile != ':memory:':
            dbfile = os.path.join(WORKSPACE, os.path.basename(dbfile))

        # you may want to pass sqlite3.deltypes or something like that
        # but at this moment I think that will make matters worse
        self.conn = sqlite3.connect(dbfile)

        # row_factory is problematic don't use it
        # you can avoid the problems but not worth it
        # if you really need performance then just use "run"
        self._cursor = self.conn.cursor()

        # some performance tuning
        self._cursor.execute(f'PRAGMA cache_size={cache_size}')
        self._cursor.execute('PRAGMA synchronous=OFF')
        self._cursor.execute('PRAGMA count_changes=0')
        # temp store at memory
        self._cursor.execute(f'PRAGMA temp_store={temp_store}')
        self._cursor.execute('PRAGMA journal_mode=OFF')

        # load some user-defined functions from helpers.py
        self.conn.create_function('isnum', -1, isnum)


    def save(self, fname, name=None, cols=None, fn=None, pkeys=None, encoding='utf-8'):
        # fname can be either a filename (csv or sas) or a sequence of Row(s)
        if isinstance(fname, str):
            if fname.endswith('.csv'):
                name, rows = fname[:-4], _csv_reel(fname, encoding)
            elif fname.endswith('.sas7bdat'):
                name, rows = fname[:-9], _sas_reel(fname)
            elif fname.endswith('.xlsx'):
                name, rows = fname[:-5], _excel_reel(fname)
            else:
                raise ValueError(f'Unknonwn File Extension {fname}')
        # Then it should be a sequence of Rows
        else:
            rows = fname

        if not name:
            raise ValueError("Table name required")

        temp_name = 'table_' + random_string(10)
        rows1 = (fn(r) for r in rows) if fn else rows

        row0, rows2 = peek_first(rows1)
        # if cols not specified row0 must be an instance of Row
        cols = listify(cols) if cols else row0.columns
        seq_values = _safe_values(rows2, cols) if isinstance(row0, Row) else rows2

        pkeys = listify(pkeys) if pkeys else None

        try:
            # you need temporary cursor.
            tempcur = self.conn.cursor()
            _sqlite3_save(tempcur, seq_values, temp_name, cols, pkeys)
        finally:
            self.rename(temp_name, name)
            tempcur.close()

    # Limited SQL is allowed cols, order, and group are all just a list of column names
    # no other SQL attages are allowed
    def apply(self, fn, tname, args=None, name=None, \
           cols='*', where=None, order=None, group=None, roll=None, pkeys=None, max_workers=None):
        def flatten(seq):
            for x in seq:
                try:
                    yield from x
                except:
                    yield x

        name1 = name or getattr(fn, '__name__')
        seq = self.reel(tname, cols, where, order, group, roll)
        args1 = (repeat(a) for a in args)
        if max_workers and max_workers > 1:
            seq1 = pmap(fn, seq, *args1, max_workers)
        else:
            seq1 = (fn(*a) for a in zip(seq, *args1))
        self.save(flatten(seq1), name1, pkeys=pkeys)


    def reel(self, tname, cols='*', where=None, order=None, group=None, roll=None):
        """Generates a sequence of rows from a query.

        query:  select statement or table name
        """
        order = listify(order) if order else []
        group = listify(group) if group else []
        dcol = None
        if roll:
            try:
                period, jump, dcol, nextfn = roll
            except:
                raise ValueError(f'Invalid parameters for rolling {roll}')
        order = ([dcol] if dcol else []) + group + \
                [c for c in order if (c not in group) and c != dcol]

        qrows = self._cursor.execute(_build_query(tname, cols, where, order))
        columns = [c[0] for c in qrows.description]
        # there can't be duplicates in column names
        if len(columns) != len(set(columns)):
            raise ValueError('duplicates in columns names')

        rows = _build_rows(qrows, columns)
        if (not group) and (not roll):
            yield from rows
        elif group and (not roll):
            for _, rs in groupby(rows, _build_keyfn(group)):
                yield Rows(rs)
        else:
            keyfn = _build_keyfn(dcol)
            for ls in _roll(rows, period, jump, keyfn, nextfn):
                # you've gone through _roll, there can't be too many iterations
                if group:
                    yield from Rows(ls).group(group, order=False)
                else:
                    yield from Rows(ls)

    # Be careful so that you don't overwrite the file
    def show(self, tname, n=None, cols='*', where=None, order=None,
             file=None, excel=False, encoding='utf-8'):
        "Printing to a screen or saving to a file "
        rows = self.reel(tname, cols=cols, where=where, order=order)
        if not file:
            # show as little if possible
            _show(rows, n or 10, cols)
            if excel:
                _open_excel(rows, None, cols, encoding)
        else:
            file = file if isinstance(file, str) else sys.stdout
            # show as much if possible
            rows  = islice(rows, n) if isinstance(n, int) and n > 0 else rows
            _csv(rows, file, cols, encoding)
            if excel:
                _open_excel(rows, file, cols, encoding)

    # register function to sql
    def register(self, fn):
        def newfn(*args):
            try:
                return fn(*args)
            except:
                return ''

        args = []
        for p in inspect.signature(fn).parameters.values():
            if p.kind != p.VAR_POSITIONAL:
                args.append(p)
        n = len(args) if args else -1
        self.conn.create_function(fn.__name__, n, newfn)

    # register aggregate function to sql
    def registerAgg(self, fn, cols=None):
        clsname = 'Temp' + random_string()
        d = {}
        if cols:
            cols = listify(cols)
            def step(self, *args):
                r = Row()
                # should I?
                # assert len(cols) == len(args)
                for a, b in zip(cols, args):
                    r[a] = b
                self.rows.append(r)
            d['step'] = step

            def finalize(self):
                rs = AggBuilder.finalize(self)
                try:
                    return fn(Rows(rs))
                except:
                    return ''

            d['finalize'] = finalize
            self.conn.create_aggregate(fn.__name__, \
                len(cols), type(clsname, (AggBuilder,), d))
        else:
            def finalize(self):
                rs = AggBuilder.finalize(self)
                try:
                    return fn(rs)
                except:
                    return ''
            d['finalize'] = finalize
            self.conn.create_aggregate(fn.__name__, -1, \
                type(clsname, (AggBuilder,), d))

    def plot(self, query, cols=None):
        self.rows(query).plot(cols)

    @property
    def tables(self):
        "list[str]: column names"
        query = self._cursor.execute("""
        select * from sqlite_master
        where type='table'
        """)
        # **.lower()
        tables = [row[1].lower() for row in query]
        return sorted(tables)


    def remdup(self, tname, name=None, cols='*', where=None, order=None, group=None, pkeys=None):
        "remove duplicates"
        cols = self._cols(_build_query(tname, cols, where, order))
        group = listify(group) if group else cols
        pkeys = listify(pkeys) if pkeys else [c for c in self._pkeys(tname) if c in cols]
        self.apply(lambda rs: rs[0], tname, name=(name or tname),
                   cols=cols, where=where, order=order, group=group, pkeys=pkeys)


    # args can be a list, a tuple or a dictionary
    # It is unlikely that we need to worry about the security issues
    # but still there's no harm. So...
    def sql(self, query):
        """Simply executes sql statement and update tables attribute

        query: SQL query string
        args: args for SQL query
        """
        return self._cursor.execute(query)



    def rows(self, tname, cols='*', where=None, order=None):
        return Rows(self.reel(tname, cols, where, order))


    def df(self, tname, cols='*', where=None, order=None):
        return self.rows(tname, cols, where, order).df(cols)


    def drop(self, tables):
        " drop table if exists "
        tables = listify(tables)
        for table in tables:
            # you can't use '?' for table name
            # '?' is for data insertion
            self.sql(f'drop table if exists {table}')

    def rename(self, old, new):
        if old.lower() in self.tables:
            self.sql(f'drop table if exists { new }')
            self.sql(f'alter table { old } rename to { new }')

    def _cols(self, query):
        return [c[0] for c in self.sql(query).description]

    def _pkeys(self, tname):
        "Primary keys in order"
        pks = [r for r in self.sql(f'pragma table_info({tname})') if r[5]]
        return [r[1] for r in sorted(pks, key=lambda r: r[5])]

    def new(self, query, name=None, pkeys=None):
        """Create new table from query
        """
        temp_name = 'table_' + random_string()
        tname = _get_name_from_query(query)
        # keep pkeys from the original table if not exists
        pkeys = listify(pkeys) if pkeys else self._pkeys(tname)
        name = name or tname
        try:
            self.run(_create_statement(temp_name, self._cols(query), pkeys))
            self.run(f'insert into {temp_name} {query}')
            self.run(f'drop table if exists { name }')
            self.run(f"alter table { temp_name } rename to { name }")
        finally:
            self.run(f'drop table if exists { temp_name }')


    def join(self, *tinfos, name=None, pkeys=None):
        "simplified version of left join"
        # if name is not given first table name
        def get_newcols(cols):
            result = []
            for c in listify(cols.lower()):
                a, *b = [x.strip() for x in c.split('as')]
                result.append(b[0] if b else a)
            return result
        name = name or tinfos[0][0]
        cols0 = get_newcols(tinfos[0][1])
        pkeys = listify(pkeys) if pkeys else [c for c in self._pkeys(tinfos[0][0]) if c in cols0]

        # Validity checks
        all_newcols = []
        mcols_sizes = []
        for _, cols, mcols in tinfos:
            all_newcols += get_newcols(cols)
            mcols_sizes.append(len(listify(mcols)))
        assert len(all_newcols) == len(set(all_newcols)), "Column duplicates"
        assert len(set(mcols_sizes)) == 1, "Matching columns must have the same sizes"
        assert all(listify(tinfos[0][2])), "First table can't have empty matching columns"
        # At least on matching column must exist but it's hard to miss

        try:
            tcols = []
            # write new temporary tables for performance
            for tname, cols, mcols in tinfos:
                if isinstance(mcols, dict):
                    temp_tname = 'table_' + random_string()
                    def build_fn(mcols):
                        def fn(r):
                            for c, f in mcols.items():
                                if f:
                                    r[c] = f(r[c])
                            return r
                    self.do(build_fn(mcols), tname, name=temp_tname, pkeys=list(mcols))
                elif self._pkeys(tname) != listify(mcols):
                    temp_tname = 'table_' + random_string()
                    self.new(tname, name=temp_tname, pkeys=listify(mcols))
                else:
                    temp_tname = tname
                newcols = [temp_tname + '.' + c for c in listify(cols)]
                tcols.append((temp_tname, newcols, listify(mcols)))

            tname0, _, mcols0 = tcols[0]
            join_clauses = []
            for tname1, mcols1 in tcols[1:]:
                eqs = []
                for c0, c1 in zip(mcols0, mcols1):
                    if c1:
                        eqs.append(f'{tname0}.{c0} = {tname1}.{c1}')
                join_clauses.append(f"""
                left join {tname1}
                on {' and '.join(eqs)}
                """)
            jcs = ' '.join(join_clauses)
            allcols = ', '.join(c for _, cols, _ in tcols for c in cols)
            query = f"select {allcols} from {tname0} {jcs}"
            self.new(query, name, pkeys)
        finally:
            # drop temporary tables
            for (t0, _, _), (t1, _, _) in zip(tinfos, tcols):
                if t0 != t1:
                    self.drop(t1)


def _safe_values(rows, cols):
    "assert all rows have cols"
    for r in rows:
        assert r.columns == cols, str(r)
        yield r.values


def _pick(cols, seq):
    " pick only cols for a seq, similar to sql select "
    cols = listify(cols)
    for r in seq:
        r1 = Row()
        for c in cols:
            r1[c] = r[c]
        yield r1


def _build_keyfn(key):
    " if key is a string return a key function "
    # if the key is already a function, just return it
    if hasattr(key, '__call__'):
        return key
    colnames = listify(key)
    if len(colnames) == 1:
        col = colnames[0]
        return lambda r: r[col]
    else:
        return lambda r: [r[colname] for colname in colnames]


def _gen_valid_column_names(columns):
    """Generate valid column names from arbitrary ones

    Note:
        Every column name is lowercased
        >>> _gen_valid_column_names(['a', '_b', 'a', 'a1"*c', 'a1c'])
        ['a0', 'a_b', 'a1', 'a1c0', 'a1c1']
    """
    # Some of the sqlite keywords are not allowed for column names
    # http://www.sqlite.org/sessions/lang_keywords.html
    sqlite_keywords = {
        "ABORT", "ACTION", "ADD", "AFTER", "ALL", "ALTER", "ANALYZE", "AND",
        "AS", "ASC", "ATTACH", "AUTOINCREMENT", "BEFORE", "BEGIN", "BETWEEN",
        "BY", "CASCADE", "CASE", "CAST", "CHECK", "COLLATE", "COLUMN",
        "COMMIT", "CONFLICT", "CONSTRAINT", "CREATE", "CROSS", "CURRENT_DATE",
        "CURRENT_TIME", "CURRENT_TIMESTAMP", "DATABASE", "DEFAULT",
        "DEFERRABLE", "DEFERRED", "DELETE", "DESC", "DETACH", "DISTINCT",
        "DROP", "EACH", "ELSE",
        "END", "ESCAPE", "EXCEPT", "EXCLUSIVE", "EXISTS", "EXPLAIN", "FAIL",
        "FOR", "FOREIGN", "FROM", "FULL", "GLOB", "GROUP", "HAVING", "IF",
        "IGNORE", "IMMEDIATE", "IN", "INDEX", "INDEXED", "INITIALLY", "INNER",
        "INSERT", "INSTEAD", "INTERSECT", "INTO", "IS", "ISNULL", "JOIN",
        "KEY", "LEFT", "LIKE", "LIMIT", "MATCH", "NATURAL",
        # no is ok somehow
        # no idea why
        # "NO",
        "NOT", "NOTNULL", "NULL", "OF", "OFFSET", "ON", "OR", "ORDER", "OUTER",
        "PLAN", "PRAGMA", "PRIMARY", "QUERY", "RAISE", "REFERENCES",
        "REGEXP", "REINDEX", "RENAME", "REPLACE", "RESTRICT", "RIGHT",
        "ROLLBACK", "ROW", "SAVEPOINT", "SELECT", "SET", "TABLE", "TEMP",
        "TEMPORARY", "THEN", "TO", "TRANSACTION",
        "TRIGGER", "UNION", "UNIQUE", "UPDATE", "USING", "VACUUM", "VALUES",
        "VIEW", "VIRTUAL", "WHEN", "WHERE",

        # These are not sqlite keywords but attribute names of Row class
        'COLUMNS', 'VALUES',
    }

    default_column_name = 'col'
    temp_columns = []
    for col in columns:
        # save only alphanumeric and underscore
        # and remove all the others
        newcol = camel2snake(re.sub(r'[^\w]+', '', col))
        if newcol == '':
            newcol = default_column_name
        elif not newcol[0].isalpha() or newcol.upper() in sqlite_keywords:
            newcol = 'a_' + newcol
        temp_columns.append(newcol)

    # no duplicates
    if len(temp_columns) == len(set(temp_columns)):
        return temp_columns

    # Tag numbers to column-names starting from 0 if there are duplicates
    cnt = {col: n for col, n in Counter(temp_columns).items() if n > 1}
    cnt_copy = dict(cnt)

    result_columns = []
    for col in temp_columns:
        if col in cnt:
            result_columns.append(col + str(cnt_copy[col] - cnt[col]))
            cnt[col] -= 1
        else:
            result_columns.append(col)
    return result_columns


def _create_statement(name, colnames, pkeys):
    """create table if not exists foo (...)

    Note:
        Every type is numeric.
        Table name and column names are all lower cased
    """
    pkeys = [f"primary key ({', '.join(pkeys)})"] if pkeys else []
    # every col is numeric, this may not be so elegant but simple to handle.
    # If you want to change this, Think again
    schema = ', '.join([col.lower() + ' ' + 'numeric' for col in colnames] + pkeys)
    return "create table if not exists %s (%s)" % (name.lower(), schema)


def _insert_statement(name, ncol):
    """insert into foo values (?, ?, ?, ...)
    Note:
        Column name is lower cased

    ncol : number of columns
    """
    qmarks = ', '.join(['?'] * ncol)
    return "insert into %s values (%s)" % (name.lower(), qmarks)


def _build_query(tname, cols=None, where=None, order=None):
    cols = ', '.join(listify(cols)) if cols else '*'
    where = 'where ' + where if where else ''
    order = 'order by ' + ', '.join(listify(order)) if order else ''
    return f'select {cols} from {tname} {where} {order}'


def _sqlite3_save(cursor, srows, table_name, column_names, pkeys):
    "saves sqlite3.Row instances to db"
    cursor.execute(_create_statement(table_name, column_names, pkeys))
    istmt = _insert_statement(table_name, len(column_names))
    try:
        cursor.executemany(istmt, srows)
    except:
        raise Exception("Trying to insert invaid Values to DB")


def _write_all(lines, file):
    "Write all to csv"
    # you need to pass newline for Windows
    w = csv.writer(file, lineterminator='\n')
    for line in lines:
        w.writerow(line)


# write to a csv
def _csv(rows, file, cols, encoding='utf-8'):
    if cols:
        rows = _pick(cols, rows)
    row0, rows1 = peek_first(rows)
    if isinstance(row0, Row):
        seq_values = chain([row0.columns], _safe_values(rows1, row0.columns))
    else:
        seq_values = rows1
    if file == sys.stdout:
        _write_all(seq_values, file)
    elif isinstance(file, str):
        try:
            fout = open(os.path.join(WORKSPACE, file), 'w', encoding=encoding)
            _write_all(seq_values, fout)
        finally:
            fout.close()
    elif isinstance(file, io.TextIOBase):
        try:
            _write_all(seq_values, file)
        finally:
            file.close()
    else:
        raise ValueError('Invalid file', file)


def _show(rows, n, cols):
    """Printing to a screen or saving to a file

    rows: iterator of Row instances
    n: maximum number of lines to show
    cols:  columns to show
    """
    # so that you can easily maintain code
    # Searching nrows is easier than searching n in editors
    nrows = n
    if cols:
        rows = _pick(cols, rows)

    row0, rows1 = peek_first(rows)
    cols = row0.columns
    seq_values = _safe_values(rows1, cols)

    with pd.option_context("display.max_rows", nrows), \
            pd.option_context("display.max_columns", 1000):
        # make use of pandas DataFrame displaying
        # islice 1 more rows than required
        # to see if there are more rows left
        list_values = list(islice(seq_values, nrows + 1))
        print(pd.DataFrame(list_values[:nrows], columns=cols))
        if len(list_values) > nrows:
            print("...more rows...")


# sequence row values to rows
def _build_rows(seq_values, cols):
    "build rows from an iterator of values"
    for vals in seq_values:
        r = Row()
        for col, val in zip(cols, vals):
            r[col] = val
        yield r


def _get_name_from_query(query):
    """'select * from foo where ...' => foo
    """
    pat = re.compile(r'\s+from\s+(\w+)\s*')
    try:
        return pat.search(query.lower()).group(1)
    except:
        return None


def _open_excel(rows, file, cols, encoding):
    def _open(file):
        filepath = os.path.join(WORKSPACE, file)
        if os.path.isfile(filepath):
            if os.name == 'nt':
                cmd = 'start excel'
            elif platform.system() == 'Linux':
                cmd = 'libreoffice'
            elif os.name == 'posix':
                cmd = 'open -a "Microsoft Excel"'
            try:
                os.system(f'{cmd} {filepath}')
            except:
                print("Excel not found")
        else:
            print(f'File does not exist {filepath}')

    if isinstance(file, str):
        _open(file)
    else:
        # non binary
        file = tempfile.NamedTemporaryFile('w', encoding=encoding, delete=True)
        _csv(rows, file, cols, encoding)
        _open(file.name)


def _roll(seq, period, jump, keyfn, nextfn):
    """generates chunks of seq for rollover tasks.
    seq is assumed to be ordered
    """
    def chunk(seq):
        fst, seq1 = peek_first(seq)
        k0 = keyfn(fst)
        for k1, sq in groupby(seq1, keyfn):
            if k0 == k1:
                k0 = nextfn(k1)
                # you must realize them first
                yield list(sq)
            else:
                # some missings
                while k0 < k1:
                    k0 = nextfn(k0)
                    yield []
                k0 = nextfn(k1)
                # you must realize them first
                yield list(sq)

    gss = tee(chunk(seq), period)
    for i, gs in enumerate(gss):
        # consume
        for i1 in range(i):
            next(gs)

    for xs in islice(zip_longest(*gss, fillvalue=None), 0, None, jump):
        # this might be a bit inefficient for some cases
        # but this is convenient, let's just go easy,
        # not making mistakes is much more important
        result =  list(chain(*(x for x in xs if x)))
        if len(result) > 0:
            yield result


# EVERY COLUMN IS A STRING!!!
# Possible the messiest type of file
def _csv_reel(csv_file, encoding):
    "Loads well-formed csv file, 1 header line and the rest is data "
    def is_empty_line(line):
        """Tests if a list of strings is empty for example ["", ""] or []
        """
        return [x for x in line if x.strip() != ""] == []

    with open(os.path.join(WORKSPACE, csv_file), encoding=encoding) as fin:
        first_line = fin.readline()[:-1]
        columns = _gen_valid_column_names(listify(first_line))
        ncol = len(columns)

        # reader = csv.reader(fin)
        # NULL byte error handling
        reader = csv.reader(x.replace('\0', '') for x in fin)
        for line_no, line in enumerate(reader, 2):
            if len(line) != ncol:
                if is_empty_line(line):
                    continue
                raise ValueError(
                    """%s at line %s column count not matched %s != %s: %s
                    """ % (csv_file, line_no, ncol, len(line), line))
            row1 = Row()
            for col, val in zip(columns, line):
                row1[col] = val
            yield row1


def _sas_reel(sas_file):
    filename = os.path.join(WORKSPACE, sas_file)
    with SAS7BDAT(filename) as f:
        reader = f.readlines()
        # lower case
        header = [x.lower() for x in next(reader)]
        for line in reader:
            r = Row()
            for k, v in zip(header, line):
                r[k] = v
            yield r


# this could be more complex but should it be?
def _excel_reel(excel_file):
    filename = os.path.join(WORKSPACE, excel_file)
    # it's OK. Excel files are small
    df = pd.read_excel(filename)
    yield from _df_reel(df, False)


# Might be exported later.
def _df_reel(df, index=True):
    cols = df.columns
    c0 = _gen_valid_column_names(['idx'] + df.columns)[0]
    for i, r in df.iterrows():
        r0 = Row()
        if index:
            r0[c0] = i
        for c, v in zip(cols, (r[c] for c in cols)):
            r0[c] = v
        yield r0


