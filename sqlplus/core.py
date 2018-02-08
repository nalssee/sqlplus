"""
Utils for statistical analysis using sqlite3 engine
"""
import os
import sqlite3
import copy
import warnings
import inspect
import numpy as np
import csv
import statistics as st
import pandas as pd
import shutil
from concurrent.futures import ProcessPoolExecutor
from sas7bdat import SAS7BDAT
from contextlib import contextmanager
from itertools import groupby, islice, chain, tee, \
    zip_longest, accumulate
from openpyxl import load_workbook

from .util import isnum, _listify, _peek_first, \
    _random_string, dmath, dconv

# pandas raises warnings because maintainers of statsmodels are lazy
warnings.filterwarnings('ignore')
import statsmodels.api as sm


WORKSPACE = ''


def setdir(path):
    """Set working directory

    Args:
        path(str): Absolute path
    """
    global WORKSPACE
    WORKSPACE = path
    if not os.path.exists(WORKSPACE):
        os.makedirs(WORKSPACE)


@contextmanager
def connect(dbfile, cache_size=100000, temp_store=2):
    # temp_store might be deprecated
    """ Connects to SQL database, decorated with contextmanager

    Usage:
        with connect('dbfile.db') as conn:
            conn.load('sample.csv')

    Args:
        |  dbfile(str): relative path for db file.
        |  cache_size(int):  cache size for insertion.
        |  temp_store(int):  0, 1 for file. 2 for memory.
    """
    splus = SQLPlus(dbfile, cache_size, temp_store)
    try:
        yield splus
    finally:
        # should I close the cursor?
        splus._cursor.close()
        splus.conn.commit()
        splus.conn.close()


# aggreate function builder
class _AggBuilder:
    def __init__(self):
        self.rows = []

    def step(self, *args):
        self.rows.append(args)


# Don't try to be smart, unless you really know well
class Row:
    """Mutable version of sqlite3.row

    Note:
        |  The order of assignment is preserved

        Row value types are one of int, float or str
    """
    # works for python 3.6 and higher
    def __init__(self, **kwargs):
        super().__setattr__('_dict', kwargs)

    @property
    def columns(self):
        """Returns a list of column names(strings)
        """
        return list(self._dict.keys())

    @property
    def values(self):
        """Returns a list of column values
        """
        return list(self._dict.values())

    def __getattr__(self, name):
        return self._dict[name]

    def __setattr__(self, name, value):
        self._dict[name] = value

    def __delattr__(self, name):
        del self._dict[name]

    def __getitem__(self, name):
        return self._dict[name]

    def __setitem__(self, name, value):
        self._dict[name] = value

    def __delitem__(self, name):
        del self._dict[name]

    def __repr__(self):
        content = ', '.join(c + '=' + repr(v) for c, v in self._dict.items())
        return 'Row(' + content + ')'

    # for pickling, very important
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    # TODO:
    # hasattr doesn't work properly
    # you can't make it work by changing getters and setters
    # to an ordinary way. but it is slower


class Rows:
    """Wrapper for a list of Row instances

    Attributes:
        rows(list of Row instances)
    """
    # don't try to define __getattr__, __setattr__
    # list objects has a lot of useful attributes that can't be overwritten
    # not the same situation as 'row' class

    # inheriting list can be problematic
    # when you want to use this as a superclass
    # see 'where' method, you must return 'self' but it's not efficient
    # (at least afaik) if you inherit list

    def __init__(self, rows):
        """
        Args:
            rows(sequence of Row instances)
        """
        self.rows = list(rows) if rows else []

    def __len__(self):
        return len(self.rows)

    # __getitem__ enables you to iterate 'Rows'
    def __getitem__(self, k):
        """
        Args:
            k: int, slice, or column name
        """
        if isinstance(k, int):
            return self.rows[k]
        if isinstance(k, slice):
            # shallow copy for non-destructive slicing
            return self._newrows(self.rows[k])

        k = _listify(k)
        if len(k) == 1:
            k0 = k[0]
            return [r[k0] for r in self.rows]
        return [[r[k0] for k0 in k] for r in self.rows]

    # Do not define __setitem__, it will get users confused

    def __delitem__(self, k):
        if isinstance(k, int) or isinstance(k, slice):
            del self.rows[k]
            return

        k = _listify(k)
        if len(k) == 1:
            k0 = k[0]
            for r in self.rows:
                del r[k0]
        else:
            for r in self.rows:
                for k0 in k:
                    del r[k0]

    def __add__(self, other):
        return self._newrows(self.rows + other.rows)

    def _newrows(self, rs):
        # copying rows and build Rows object
        # Am I worring too much?, this is for inheritance
        self.rows, temp = [], self.rows
        other = copy.copy(self)
        other.rows, self.rows = list(rs), temp
        return other

    # Limited version of __setitem__
    def set(self, k, v):
        for r in self.rows:
            r[k] = v

    def append(self, r):
        self.rows.append(r)
        return self

    # destructive!!!
    def order(self, key, reverse=False):
        """Order rows by key

        Args:
            |  key(str, list of str, fn)
            |  reverse(bool)

        """
        # You can pass fn as key arg but not recommended
        self.rows.sort(key=_build_keyfn(key), reverse=reverse)
        return self

    def where(self, pred):
        """Filters rows

        Args:
            pred(str or fn): "year > 1990 and size < 100" or predicate
        """
        return self._newrows([r for r in self if pred(r)])

    def isnum(self, *cols):
        """Filters rows that are all numbers in cols
        """
        cols = _listify(','.join(cols))
        return self._newrows([r for r in self if isnum(*(r[c] for c in cols))])

    def avg(self, col, wcol=None, ndigits=None):
        """Computes average

        Args:
            |  col(str): column name to compute average
            |  wcol(str): column for weight
            |  n(int): round digits

        Returns float
        """
        val = np.average(self[col], weights=self[wcol] if wcol else None)
        return round(val, ndigits) if ndigits else val

    def ols(self, model):
        """OLS fit

        Args:
            model(str): 'col1 ~ col2 + col3'

        Returns:
            |  statsmodels.api.OLS(model).fit()
            |  http://www.statsmodels.org/stable/index.html

        Note:
            Constant is added automatically
        """
        def parse_model(model):
            "y ~ x1 + x2 => ['y', 'x1', 'x2']"
            left, right = model.split('~')
            return [left.strip()] + [x.strip() for x in right.split('+')]

        y, *xs = parse_model(model)
        df = self.df()
        res = sm.OLS(df[[y]], sm.add_constant(df[xs])).fit()
        return res

    def truncate(self, col, limit=0.01):
        """Truncate rows that are out of limits

        Args:
            |  col(str): column name
            |  limit(float): for both sides respectably.

        Returns self
        """
        xs = self[col]
        lower = np.percentile(xs, limit * 100)
        higher = np.percentile(xs, (1 - limit) * 100)
        return self.where(lambda r: r[col] >= lower and r[col] <= higher)

    def winsorize(self, col, limit=0.01):
        """Winsorsize rows that are out of limits

        Args:
            |  col(str): column name.
            |  limit(float): for both sides respectably.

        Returns self
        """
        xs = self[col]
        lower = np.percentile(xs, limit * 100)
        higher = np.percentile(xs, (1 - limit) * 100)
        for r in self.rows:
            if r[col] > higher:
                r[col] = higher
            elif r[col] < lower:
                r[col] = lower
        return self

    # implicit ordering
    def group(self, key):
        """Yields rows of each group

        Args:
            key(str or list of str or fn): columnn name or fn to group

        Returns: list of Rows
        """
        # key can be a fn but not recommended
        keyfn = _build_keyfn(key)
        self.order(keyfn)
        return [self._newrows(list(rs)) for _, rs in groupby(self, keyfn)]

    def overlap(self, size, step=1, group=None):
        if group:
            xs = self.group(group)
            result = []
            for i in range(0, len(xs), step):
                result.append(self._newrows(chain(*xs[i:i + size])))
            return result
        else:
            result = []
            for i in range(0, len(self), step):
                result.append(self[i:i + size])
            return result

    def chunk(self, n, col=None):
        """Yields Rows, useful for building portfolios

        Usage:
            |  self.chunk(3) => yields 3 rows about the same size
            |  self.chunk([0.3, 0.4, 0.3]) => yields 3 rows of 30%, 40%, 30%
            |  self.chunk([100, 500, 1000], 'col')
            |      => yields 4 rows with break points 100, 500, 1000
        """
        size = len(self)
        if isinstance(n, int):
            start = 0
            result = []
            for i in range(1, n + 1):
                end = int((size * i) / n)
                # must yield anyway
                result.append(self[start:end])
                start = end
            return result
        # n is a list of percentiles
        elif not col:
            # then it is a list of percentiles for each chunk
            assert sum(n) <= 1, f"Sum of percentils for chunks must be <= 1.0"
            ns = [int(x * size) for x in accumulate(n)]
            result = []
            for a, b in zip([0] + ns, ns):
                result.append(self[a:b])
            return result
        # n is a list of break points
        else:
            self.order(col)
            start, end = 0, 0
            result = []
            for bp in n:
                while (self[end][col] < bp) and end < size:
                    end += 1
                result.append(self[start:end])
                start = end
            result.append(self[end:])
            return result

    # Use this when you need to see what's inside
    # for example, when you want to see the distribution of data.
    def df(self, cols=None):
        """Returns pandas data frame
        """
        if cols:
            cols = _listify(cols)
            return pd.DataFrame([[r[col] for col in cols] for r in self.rows],
                                columns=cols)
        else:
            cols = self.rows[0].columns
            return pd.DataFrame([r.values for r in self], columns=cols)


class SQLPlus:
    def __init__(self, dbfile, cache_size, temp_store):
        """
        Args:
            |  dbfile (str): db filename or ':memory:'
            |  cache_size(int)
            |  temp_store(int)

        Additional Functions in SQL:
            dconv, dmath, isnum
        """
        global WORKSPACE

        # set workspace if it's not there
        if not WORKSPACE:
            # default workspace
            WORKSPACE = os.getcwd()

        if dbfile != ':memory:':
            dbfile = os.path.join(WORKSPACE, dbfile)

        # you may want to pass sqlite3.deltypes or something like that
        # but at this moment I think that will make matters worse
        self.conn = sqlite3.connect(dbfile)
        # You can safely uncomment the following line
        self.conn.row_factory = sqlite3.Row

        # default cursor
        self._cursor = self.conn.cursor()
        # some performance tuning
        self._cursor.execute(f'PRAGMA cache_size={cache_size}')

        # Don't be too greedy, comment out the following line.
        # It's too dangerous. Your db file can be corrupted
        # self._cursor.execute('PRAGMA synchronous=OFF')

        self._cursor.execute('PRAGMA count_changes=0')
        # temp store at memory
        self._cursor.execute(f'PRAGMA temp_store={temp_store}')
        self._cursor.execute('PRAGMA journal_mode=OFF')

        # load some user-defined functions from util.py, istext unnecessary
        self.conn.create_function('isnum', -1, isnum)
        self.conn.create_function('dmath', 3, dmath)
        self.conn.create_function('dconv', 3, dconv)

    def fetch(self, tname, cols=None, where=None,
              order=None, group=None, overlap=None):
        """Generates a sequence of rows from a table.

        Args:
            |  tname(str): table name
            |  cols(str or list of str): columns to fetch
            |  where(str): where clause in SQL query
            |  order(str or list of str): comma separated str
            |  group(str or list of str): comma separated str
            |  overlap: int or (int, int)

        Yields:
            Row or Rows
        """

        def _overlap(seq, size, step):
            """generates chunks of seq for rollover tasks.
            seq is assumed to be ordered
            """

            ss = tee(seq, size)
            # consume
            for i, s in enumerate(ss):
                for i1 in range(i):
                    next(s)
            xss = zip_longest(*ss, fillvalue=None)
            for xs in islice(xss, 0, None, step):
                # lets just go easy
                yield [x for x in xs if x is not None]

        order = _listify(order) if order else []
        group = _listify(group) if group else []

        order = group + order

        qry = _build_query(tname, cols, where, order)
        qrows = self.conn.cursor().execute(qry)
        columns = [c[0] for c in qrows.description]
        # there can't be duplicates in column names
        if len(columns) != len(set(columns)):
            raise ValueError('Duplicates in columns names')

        if overlap:
            size, step = (overlap, 1) if isnum(overlap) else overlap
            if group:
                grows = (_build_rows(rs, columns)
                         for _, rs in groupby(qrows, _build_keyfn(group)))
                yield from (Rows(chain(*xs))
                            for xs in _overlap(grows, size, step))
            else:
                yield from (_build_rows(rs, columns)
                            for rs in _overlap(qrows, size, step))
        elif group:
            gby = groupby(qrows, _build_keyfn(group))
            yield from (_build_rows(rs, columns) for _, rs in gby)

        else:
            yield from (_build_row(r, columns) for r in qrows)

    def insert(self, rs, name, pkeys=None):
        """Insert Rows or sequence of Row(s)

        Args:
            |  rs(Rows, or sequence of Row(s))
            |  name(str): table name
            |  pkeys(str or list of str): primary keys
        """
        self.drop(name)
        rs = iter(rs)

        try:
            r0, rs = _peek_first(rs)
        except StopIteration:
            return

        cols = r0.columns
        n = len(cols)

        self._cursor.execute(_create_statement(name, cols, pkeys))
        istmt = _insert_statement(name, n)
        self._cursor.executemany(istmt, (r.values for r in rs))

    def ins(self, rs, name, pkeys=None):
        def _prep(r0):
            if name not in self.get_tables():
                self._cursor.execute(_create_statement(name, r0.columns,
                                     pkeys))
            return _insert_statement(name, len(r0.columns))

        if isinstance(rs, Row):
            self._cursor.execute(_prep(rs), rs.values)
        # list or Rows
        else:
            self._cursor.executemany(_prep(rs[0]), (r.values for r in rs))

    def load(self, filename, name=None, encoding='utf-8',
             fn=None, pkeys=None):
        """Read data file and save it on database

        Args:
            |  filename(str): .csv, .xlsx, .sas7bdat
            |  name(str): table name
            |  encoding(str): file encoding
            |  fn(Row -> Row): Row transformer
            |  pkeys(str or list of str): primary keys
        """
        if isinstance(filename, str):
            fname, ext = os.path.splitext(filename)

            if ext == '.csv':
                seq = _read_csv(filename, encoding)
            elif ext == '.xlsx':
                seq = _read_excel(filename)
            elif ext == '.sas7bdat':
                seq = _read_sas(filename)
            else:
                raise ValueError('Unknown file extension', ext)
            name = name or fname
        else:
            seq = filename

        if name in self.get_tables():
            return
        if fn:
            seq = (fn(r) for r in seq)

        self.insert(seq, name, pkeys)

    def to_csv(self, tname, outfile=None, cols=None,
               where=None, order=None, encoding='utf-8'):
        """Table to csv file

        Args:
            |  tname(str): table name
            |  outfile(str): output file(csv) name
            |  cols(str or list of str)
            |  where(str)
            |  order(str)
            |  encoding(str)
        """
        seq = self.fetch(tname, cols=cols, where=where, order=order)
        r0, rs = _peek_first(seq)
        columns = _listify(cols) if cols else r0.columns
        filename = outfile or tname + '.csv'
        with open(os.path.join(WORKSPACE, filename), 'w', newline='',
                  encoding=encoding) as f:
            w = csv.writer(f, delimiter=',')
            w.writerow(columns)
            for r in rs:
                w.writerow(r.values)

    # register function to sql
    def register(self, fn, name=None):
        """Register Python function on SQL
        """
        def newfn(*args):
            try:
                return fn(*args)
            # Whatever is fine
            except Exception:
                return ''

        args = []
        for p in inspect.signature(fn).parameters.values():
            if p.kind != p.VAR_POSITIONAL:
                args.append(p)
        n = len(args) if args else -1
        name = name or fn.__name__
        self.conn.create_function(name, n, newfn)

    # register aggregate function to sql
    def register_agg(self, fn, name=None):
        """Register aggregate Python function on SQL

        Args:
            |  fn: each arg is a list of column values
            |  name(str): function name to use in SQL
        """
        d = {}

        def finalize(self):
            try:
                return fn(*(x for x in zip(*self.rows)))
            except Exception:
                return ''
        d['finalize'] = finalize

        args = []
        for p in inspect.signature(fn).parameters.values():
            if p.kind != p.VAR_POSITIONAL:
                args.append(p)
        n = len(args) if args else -1

        clsname = 'Temp' + _random_string()
        name = name or fn.__name__
        self.conn.create_aggregate(fn.__name__, n,
                                   type(clsname, (_AggBuilder,), d))

    def get_tables(self):
        """List of table names in database
        """
        query = self._cursor.execute("""
        select * from sqlite_master
        where type='table'
        """)
        tables = [row[1] for row in query]
        return tables

    # args can be a list, a tuple or a dictionary
    # It is unlikely that we need to worry about the security issues
    # but still there's no harm. So...
    def sql(self, query):
        """Simply executes sql statement and update tables attribute

        Args:
            |  query(str): SQL query
        """
        return self._cursor.execute(query)

    def rows(self, tname, cols=None, where=None, order=None):
        """Returns Rows
        """
        return Rows(self.fetch(tname, cols, where, order))

    def df(self, tname, cols=None, where=None, order=None):
        """Returns pandas data frame
        """
        return self.rows(tname, cols, where, order).df(cols)

    def drop(self, tables):
        """Drops tables if exist

        Args:
            tables(str, list of str)
        """
        tables = _listify(tables)
        for table in tables:
            # you can't use '?' for table name
            # '?' is for data insertion
            self.sql(f'drop table if exists {table}')

    def rename(self, old, new):
        """Rename a table from old to new
        """
        if old in self.get_tables():
            self.sql(f'drop table if exists { new }')
            self.sql(f'alter table { old } rename to { new }')

    def create(self, query, name=None, pkeys=None):
        """Create new table from query(select statement)

        Args:
            |  query(str)
            |  name(str): new table name, original table from the query
            |             if not exists
            |  pkeys(str or list of str): primary keys
        """
        temp_name = 'table_' + _random_string()
        idx = query.lower().split().index('from')
        tname = query.split()[idx + 1]
        # keep pkeys from the original table if not exists
        pkeys = _listify(pkeys) if pkeys else self._pkeys(tname)
        name = name or tname
        try:
            self.sql(_create_statement(temp_name, self._cols(query), pkeys))
            self.sql(f'insert into {temp_name} {query}')
            self.sql(f'drop table if exists { name }')
            self.sql(f"alter table { temp_name } rename to { name }")
        finally:
            self.sql(f'drop table if exists { temp_name }')

    def join(self, *tinfos, name=None, pkeys=None):
        """Simplified version of left join

        Args:
            |  tinfo: [tname, cols, cols to match]
            |         ex) ['sample', 'col1, col2 as colx', 'col3, col4']
            |  name(str): new table name
            |             the first table name of tinfos
            |  pkeys(str or list of str): primary keys
        """
        def get_newcols(cols):
            # extract new column names
            # if there's any renaming
            result = []
            for c in _listify(cols):
                a, *b = [x.strip() for x in c.split('as')]
                result.append(b[0] if b else a)
            return result

        # No name specified, then the first table name is the output table name
        name = name or tinfos[0][0]
        # rewrite tinfos if there's missing matching columns
        mcols0 = tinfos[0][2]

        # validity check, what if mcols is a list of more than 1 element?
        for x in tinfos:
            assert len(x) == 2 or len(x) == 3, f"Invalid Arg: {x}"
        tinfos = [[tname, cols, mcols[0] if mcols else mcols0]
                  for tname, cols, *mcols in tinfos]

        # Validity checks
        all_newcols = []
        # number of matching columns for each given table
        mcols_sizes = []
        for _, cols, mcols in tinfos:
            all_newcols += get_newcols(cols)
            mcols_sizes.append(len(_listify(mcols)))

        assert len(all_newcols) == len(set(all_newcols)), "Column duplicates"
        assert len(set(mcols_sizes)) == 1,\
            "Matching columns must have the same sizes"

        tcols = []
        # write new temporary tables for performance
        for tname, cols, mcols in tinfos:
            newcols = [tname + '.' + c for c in _listify(cols)]
            tcols.append((tname, newcols, _listify(mcols)))

        tname0, _, mcols0 = tcols[0]
        join_clauses = []
        for tname1, _, mcols1 in tcols[1:]:
            eqs = []
            for c0, c1 in zip(mcols0, mcols1):
                if c1:
                    eqs.append(f'{tname0}.{c0} = {tname1}.{c1}')
            join_clauses.append(f" left join {tname1} on {' and '.join(eqs)} ")
        jcs = ' '.join(join_clauses)
        allcols = ', '.join(c for _, cols, _ in tcols for c in cols)
        query = f"select {allcols} from {tname0} {jcs}"
        self.create(query, name, pkeys)

    def pwork(self, fn, tname, args):
        n = len(args)
        rndstr = _random_string()
        tempdbs = ['temp' + rndstr + str(i) for i in range(n)]
        pkeys = self._pkeys(tname)
        try:
            with connect(tempdbs[0]) as c:
                c.insert(self.fetch(tname), tname, pkeys=pkeys)
            for tempdb in tempdbs[1:]:
                shutil.copyfile(os.path.join(WORKSPACE, tempdbs[0]),
                                os.path.join(WORKSPACE, tempdb))

            with ProcessPoolExecutor(max_workers=n) as exe:
                exe.map(fn, tempdbs, args)

            with connect(tempdbs[0]) as c:
                tables = [t for t in c.get_tables() if t != tname]

            for table in tables:
                self._collect(table, tempdbs)
        finally:
            for tempdb in tempdbs:
                fname = os.path.join(WORKSPACE, tempdb)
                if os.path.isfile(fname):
                    os.remove(fname)

    def _collect(self, tname, dbnames):
        with connect(dbnames[0]) as c:
            cols = c._cols(f"select * from {tname}")
            pkeys = c._pkeys(tname)

        self.drop(tname)
        self.sql(_create_statement(tname, cols, pkeys))
        ismt = _insert_statement(tname, len(cols))
        for dbname in dbnames:
            with connect(dbname) as c:
                self._cursor.executemany(ismt,
                                         (r.values for r in c.fetch(tname)))

    def _cols(self, query):
        return [c[0] for c in self.sql(query).description]

    def _pkeys(self, tname):
        "Primary keys in order"
        pks = [r for r in self.sql(f'pragma table_info({tname})') if r[5]]
        return [r[1] for r in sorted(pks, key=lambda r: r[5])]


def _build_keyfn(key):
    " if key is a string return a key function "
    # if the key is already a function, just return it
    if hasattr(key, '__call__'):
        return key
    colnames = _listify(key)
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
    pkeys = [f"primary key ({', '.join(_listify(pkeys))})"] if pkeys else []
    # every col is numeric, this may not be so elegant but simple to handle.
    # If you want to change this, Think again
    schema = ', '.join([col + ' ' + 'numeric' for col in colnames] + pkeys)
    return "create table if not exists %s (%s)" % (name, schema)


def _insert_statement(name, ncol):
    """insert into foo values (?, ?, ?, ...)
    """
    qmarks = ', '.join(['?'] * ncol)
    return "insert into %s values (%s)" % (name, qmarks)


def _build_query(tname, cols=None, where=None, order=None):
    "Build select statement"
    cols = ', '.join(_listify(cols)) if cols else '*'
    where = 'where ' + where if where else ''
    order = 'order by ' + ', '.join(_listify(order)) if order else ''
    return f'select {cols} from {tname} {where} {order}'


# sequence row values to rows
def _build_rows(qrows, cols):
    return Rows([_build_row(qr, cols) for qr in qrows])


def _build_row(qr, cols):
    r = Row()
    for c, v in zip(cols, qr):
        r[c] = v
    return r


def _read_csv(filename, encoding='utf-8'):
    "Loads well-formed csv file, 1 header line and the rest is data "
    def is_empty_line(line):
        """Tests if a list of strings is empty for example ["", ""] or []
        """
        return [x for x in line if x.strip() != ""] == []

    with open(os.path.join(WORKSPACE, filename),
              encoding=encoding) as fin:
        first_line = fin.readline()[:-1]
        columns = _listify(first_line)
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
                    """ % (filename, line_no, ncol, len(line), line))
            yield _build_row(line, columns)


def _read_sas(filename):
    filename = os.path.join(WORKSPACE, filename)
    with SAS7BDAT(filename) as f:
        reader = f.readlines()
        # lower case
        header = next(reader)
        for line in reader:
            yield _build_row(line, header)


# this could be more complex but should it be?
def _read_excel(filename):
    def read_df(df):
        cols = df.columns
        for i, r in df.iterrows():
            yield _build_row((str(r[c]) for c in cols), cols)

    filename = os.path.join(WORKSPACE, filename)
    # it's OK. Excel files are small
    df = pd.read_excel(filename)
    yield from read_df(df)


def readxl(fname, sheet_name='Sheet1'):
    fname = os.path.join(WORKSPACE, fname)
    result = []
    workbook = load_workbook(fname)
    for row in workbook[sheet_name].iter_rows():
        result.append([c.value or '' for c in row])
    return result

