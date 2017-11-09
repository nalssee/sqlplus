"""
sqlite3 based utils for statistical analysis

reeling off rows from db(sqlite3) and saving them back to db
"""
import os
import sys
import csv
import re
import sqlite3
import copy
import warnings
import inspect
import platform
# import operator
import numpy as np
import matplotlib
# You need to specify in macos somehow
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import statistics as st
import pandas as pd
from pandas.plotting import scatter_matrix

from collections import OrderedDict
from contextlib import contextmanager
from itertools import groupby, islice, chain, tee, \
    zip_longest, repeat
from pypred import Predicate

from .util import isnum, listify, peek_first, \
    parse_model, random_string, pmap

# pandas raises warnings because maintainers of statsmodels are lazy
warnings.filterwarnings('ignore')
import statsmodels.api as sm

__all__ = ['dbopen', 'Row', 'Rows']


WORKSPACE = ''
inf = float('inf')
epsilon = sys.float_info.epsilon


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
    def __getitem__(self, k):
        "cols: integer or list of strings or comma separated string"
        if isinstance(k, int):
            return self.rows[k]
        if isinstance(k, slice):
            # shallow copy for non-destructive slicing
            return self._newrows(self.rows[k])
        # Now k is a column name
        return [r[k] for r in self.rows]

    def __setitem__(self, k, v):
        if isinstance(k, int) or isinstance(k, slice):
            self.rows[k] = v
            return

        # same value is assigned to them all
        if not isinstance(v, list):
            for r in self.rows:
                r[k] = v
        else:
            assert len(self) == len(v), "Invalid assignment"
            for r, v1 in zip(self.rows, v):
                r[k] = v1

    def __delitem__(self, k):
        if isinstance(k, int) or isinstance(k, slice):
            del self.rows[k]
            return

        for r in self.rows:
            del r[k]

    def __add__(self, other):
        return self._newrows(self.rows + other.rows)

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

    def roll(self, period, jump, dcol, nextfn):
        "group rows over time, allowing overlaps"
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

    def where(self, pred):
        if isinstance(pred, str):
            obj = Predicate(pred)
            return self._newrows([r for r in self
                                  if obj.evaluate(r._ordered_dict)])
        return self._newrows([r for r in self if pred(r)])

    def corr(self, cols=None):
        "Lower left: Pearson, Upper right: Spearman"
        cols = cols or self[0].columns
        df = self.df(cols)
        corr1 = df.corr()
        corr2 = df.corr('spearman')
        columns = list(corr1.columns.values)
        c0 = '_'

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
        return self._newrows([r for r in self if isnum(*(r[c] for c in cols))])

    # not used often
    def istext(self, *cols):
        "another simplified filtering, texts(string) only"
        cols = listify(','.join(cols))
        return self._newrows([r for r in self
                              if all(isinstance(r[c], str) for c in cols)])

    def avg(self, col, wcol=None):
        # wcol: column for weight
        if wcol:
            rs = self.isnum(col, wcol)
            total = sum(r[wcol] for r in rs)
            return sum(r[col] * r[wcol] / total for r in rs)
        else:
            return st.mean(r[col] for r in self if isnum(r[col]))

    def ols(self, model, rows=True):
        y, *xs = parse_model(model)
        X = [[r[x] for x in xs] for r in self]
        res = sm.OLS(self[y], sm.add_constant(X)).fit()
        if rows:
            rs = []
            for i, param in enumerate(['const'] + xs):
                r = Row(param=param)
                r.coef = res.params[i]
                r.stderr = res.bse[i]
                r.tval = res.tvalues[i]
                r.pval = res.pvalues[i]
                rs.append(r)
            return Rows(rs)
        return res

    def plot(self, cols=None):
        cols = listify(cols) if cols else self[0].columns
        scatter_matrix(self.isnum(*cols).df(cols))
        plt.show()

    def truncate(self, col, limit=0.01):
        "Truncate extreme values, defalut 1 percent on both sides"
        xs = self[col]
        lower = np.percentile(xs, limit * 100)
        higher = np.percentile(xs, (1 - limit) * 100)
        self.rows = self.where(lambda r: r[col] >= lower and r[col] <= higher)\
                        .rows
        return self

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
            yield self._newrows(list(rs))

    def show(self, n=None, cols=None, file=None, encoding='utf-8',
             excel=False):
        if file or excel:
            if (isinstance(n, int) and n > 0):
                rows = self.rows[:n]
            else:
                rows = self.rows

            if file:
                _csv(rows, file, cols, encoding)
            if excel:
                _open_excel(rows, file, cols, encoding, excel)
        else:
            _show(self.rows, n or 10, cols)

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

    def pn(self, col, bps, pncol=None):
        "Assign portfolio number using the given breakpoints"
        def loc(x, bps):
            for i, b in enumerate(bps):
                if x < b:
                    return i + 1
            return len(bps) + 1

        if not pncol:
            pncol = 'pn_' + col
        self[pncol] = ''
        for r in self.rows:
            r[pncol] = loc(r[col], bps)
        return self


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

    def read(self, tname, cols=None, where=None, order=None,
             group=None, roll=None):
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
                    yield from Rows(ls).group(group)
                else:
                    yield Rows(ls)

    def write(self, seq, name, cols=None, pkeys=None):
        """
        """
        def flatten(seq):
            for x in seq:
                try:
                    yield from x
                except:
                    yield x

        temp_name = 'table_' + random_string(10)
        seq1 = (r for r in flatten(seq))

        row0, seq2 = peek_first(seq1)
        # if cols not specified row0 must be an instance of Row
        cols = listify(cols) if cols else row0.columns
        seq_values = _safe_values(seq2, cols) \
            if isinstance(row0, Row) else seq2

        pkeys = listify(pkeys) if pkeys else None

        try:
            # you need temporary cursor.
            tempcur = self.conn.cursor()
            _sqlite3_save(tempcur, seq_values, temp_name, cols, pkeys)
        finally:
            self.rename(temp_name, name)
            tempcur.close()

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
            self.conn.create_aggregate(fn.__name__, len(cols),
                                       type(clsname, (AggBuilder,), d))
        else:
            def finalize(self):
                rs = AggBuilder.finalize(self)
                try:
                    return fn(rs)
                except:
                    return ''
            d['finalize'] = finalize
            self.conn.create_aggregate(fn.__name__, -1,
                                       type(clsname, (AggBuilder,), d))

    def plot(self, tname, cols=None, where=None):
        self.rows(tname, cols=cols, where=where).plot(cols)

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

    # args can be a list, a tuple or a dictionary
    # It is unlikely that we need to worry about the security issues
    # but still there's no harm. So...
    def sql(self, query):
        """Simply executes sql statement and update tables attribute

        query: SQL query string
        args: args for SQL query
        """
        return self._cursor.execute(query)

    def rows(self, tname, cols=None, where=None, order=None):
        return Rows(self.read(tname, cols, where, order))

    def df(self, tname, cols=None, where=None, order=None):
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
        """Create new table from query(select statement)
        """
        temp_name = 'table_' + random_string()
        tname = _get_name_from_query(query)
        # keep pkeys from the original table if not exists
        pkeys = listify(pkeys) if pkeys else self._pkeys(tname)
        name = name or tname
        try:
            self.sql(_create_statement(temp_name, self._cols(query), pkeys))
            self.sql(f'insert into {temp_name} {query}')
            self.sql(f'drop table if exists { name }')
            self.sql(f"alter table { temp_name } rename to { name }")
        finally:
            self.sql(f'drop table if exists { temp_name }')

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
        # TODO: Not soure about this
        pkeys = listify(pkeys)\
            if pkeys else [c for c in self._pkeys(tinfos[0][0]) if c in cols0]

        # Validity checks
        all_newcols = []
        mcols_sizes = []
        for _, cols, mcols in tinfos:
            all_newcols += get_newcols(cols)
            mcols_sizes.append(len(listify(mcols)))

        assert len(all_newcols) == len(set(all_newcols)), "Column duplicates"
        assert len(set(mcols_sizes)) == 1,\
            "Matching columns must have the same sizes"
        assert all(listify(tinfos[0][2])), \
            "First table can't have empty matching columns"
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
                        return fn

                    fn1 = build_fn(mcols)
                    temp_seq = (fn1(r) for r in self.read(tname))
                    self.write(temp_seq, temp_tname, pkeys=list(mcols))

                else:
                    temp_tname = tname
                newcols = [temp_tname + '.' + c for c in listify(cols)]
                tcols.append((temp_tname, newcols, listify(mcols)))

            tname0, _, mcols0 = tcols[0]
            join_clauses = []
            for tname1, _, mcols1 in tcols[1:]:
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


def _create_statement(name, colnames, pkeys):
    """create table if not exists foo (...)

    Note:
        Every type is numeric.
        Table name and column names are all lower cased
    """
    pkeys = [f"primary key ({', '.join(pkeys)})"] if pkeys else []
    # every col is numeric, this may not be so elegant but simple to handle.
    # If you want to change this, Think again
    schema = ', '.join([col.lower() + ' ' + 'numeric' for col in colnames] +
                       pkeys)
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
    "Build select statement"
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
    else:
        try:
            _write_all(seq_values, file)
        finally:
            file.close()


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


def _open_excel(rows, file, cols, encoding, excel):
    # TODO: Creates an unnecessary temporary csv file
    # Take care of it. If you think it bugs you
    def _open(file):
        filepath = os.path.join(WORKSPACE, file)
        if os.path.isfile(filepath):
            if os.name == 'nt':
                excel1 = excel if isinstance(excel, str) else 'excel'
                cmd = f'start {excel1}'
            elif platform.system() == 'Linux':
                # Libreoffice calc is the only viable option for linux
                excel1 = excel if isinstance(excel, str) else 'libreoffice'
                cmd = excel1
            elif os.name == 'posix':
                # For OS X, use Numbers, not Excel.
                # It is free and good enough for this purpose.
                excel1 = excel if isinstance(excel, str) else 'numbers'
                cmd = f'open -a {excel1}'
            try:
                os.system(f'{cmd} {filepath}')
            except:
                print(f"{excel} not found")
        else:
            print(f'File does not exist {filepath}')

    if isinstance(file, str):
        _open(file)
    else:
        # A bit naive
        for f in os.listdir(WORKSPACE):
            if f.startswith('temp_') and f.endswith('.csv'):
                os.remove(os.path.join(WORKSPACE, f))
        file = 'temp_' + random_string() + '.csv'
        _csv(rows, file, cols, encoding)
        _open(file)


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
        result = list(chain(*(x for x in xs if x)))
        if len(result) > 0:
            yield result

