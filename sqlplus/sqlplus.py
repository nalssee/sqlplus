"""main file."""

import csv
import locale
import os
import random
import signal
import sqlite3
import string
from contextlib import contextmanager
from inspect import signature
from itertools import groupby
from shutil import copyfile
from pathos.multiprocessing import ProcessingPool as Pool

import pandas as pd
import psutil
from graphviz import Digraph
from more_itertools import chunked, spy
from openpyxl import Workbook
from sas7bdat import SAS7BDAT
from tqdm import tqdm

from .config import (_CONFIG, _CONN, _DBNAME, _GRAPH_NAME,
                     _JOBS, _TEMP, _WS,
                     _RESERVED_KEYWORDS)
from .exceptions import (InvalidColumns, InvalidGroup, NoRowToInsert,
                         NoRowToWrite, NoSuchTableFound, ReservedKeyword,
                         SkipThisTurn, TableDuplication, UnknownConfig)
from .logging import logger
from .util import _build_keyfn, listify, step

# from pathos.multiprocessing import ProcessingPool as Pool


@contextmanager
def _connect(dbfile):
    conn = _Connection(dbfile)
    try:
        yield conn
    finally:
        # Trying to make closing atomic to handle multiple ctrl-cs
        # Imagine the first ctrl-c have the process enter the 'finally block'
        # and the second ctrl-c interrupts the block in the middle
        # so that the database is corrupted
        with _delayed_keyboard_interrupts():
            # should I close the cursor?
            conn._cursor.close()
            conn._conn.commit()
            conn._conn.close()
            conn._is_connected = False


@contextmanager
def _delayed_keyboard_interrupts():
    signal_received = []

    def handler(sig, frame):
        nonlocal signal_received
        signal_received = (sig, frame)
        logger.debug('SIGINT received. Delaying KeyboardInterrupt.')
    old_handler = signal.signal(signal.SIGINT, handler)

    try:
        yield
    finally:
        signal.signal(signal.SIGINT, old_handler)
        if signal_received:
            old_handler(*signal_received)


class _Connection:
    def __init__(self, dbfile):
        dbfile = os.path.join(_WS[0], dbfile)
        locale.setlocale(locale.LC_ALL, _CONFIG['locale'])
        logger.propagate = _CONFIG['msg']
        self._conn = sqlite3.connect(dbfile)
        self._conn.row_factory = _dict_factory
        self._cursor = self._conn.cursor()
        self._is_connected = True
        # DO NOT re_CONFIGure pragmas. Defaults are defaults for a reason.
        # You could make it faster but with a cost. It could corrupt the disk
        # image of the database.

    def fetch(self, query, by=None):
        if by and isinstance(by, list) and by != ['*'] and\
                all(isinstance(c, str) for c in by):
            query += " order by " + ','.join(by)
        if by:
            if isinstance(by, list):
                rows = self._conn.cursor().execute(query)
                rows1 = (list(rs) for _, rs in
                         groupby(rows, _build_keyfn(by)))

            elif isinstance(by, int):
                rows = self._conn.cursor().execute(query)
                rows1 = chunked(rows, by)

            else:
                raise InvalidGroup(by)

            yield from rows1
        else:
            rows = self._conn.cursor().execute(query)
            yield from rows

    def insert(self, rs, name):
        r0, rs = spy(rs)
        if r0 == []:
            raise NoRowToInsert(name)

        cols = list(r0[0])
        for x in [name] + cols:
            if _is_reserved(x):
                raise ReservedKeyword(x)

        try:
            self._cursor.execute(_create_statement(name, cols))
            istmt = _insert_statement(name, r0[0])
            self._cursor.executemany(istmt, rs)
        except sqlite3.OperationalError:
            raise InvalidColumns(cols)

    def load(self, filename, name, delimiter=None, quotechar='"',
             encoding='utf-8', newline="\n", fn=None):
        total = None
        if isinstance(filename, str):
            _, ext = os.path.splitext(filename)
            if ext.lower() == '.xlsx' or ext.lower() == ".xls":
                seq = _read_excel(filename)
            elif ext.lower() == '.sas7bdat':
                seq = _read_sas(filename)
            elif ext.lower() == ".dta":
                seq = _read_stata(filename)
            else:
                # default delimiter is ","
                delimiter = delimiter or\
                    ("\t" if ext.lower() == ".tsv" else ",")
                seq = _read_csv(filename, delimiter=delimiter,
                                quotechar=quotechar, encoding=encoding,
                                newline=newline)
                total = _line_count(filename, encoding, newline)
        else:
            # iterator, since you can pass an iterator
            # functions of 'load' should be limited
            seq = filename

        seq = tqdm(seq, total=total)

        if fn:
            seq = _flatten(fn(rs) for rs in seq)
        self.insert(seq, name)

    def get_tables(self):
        query = self._cursor.\
            execute("select * from sqlite_master where type='table'")
        return [row['name'] for row in query]

    def drop(self, tables):
        tables = listify(tables)
        for table in tables:
            if _is_reserved(table):
                raise ReservedKeyword(table)
            self._cursor.execute(f'drop table if exists {table}')

    def join(self, tinfos, name):
        # check if a colname is a reserved keyword
        newcols = []
        for _, _cols, _ in tinfos:
            _cols = [c.upper() for c in listify(_cols)]
            for c in _cols:
                if 'AS' in c:
                    newcols.append(c.split('AS')[-1])
        for x in [name] + newcols:
            if _is_reserved(x):
                raise ReservedKeyword(x)

        tname0, _, mcols0 = tinfos[0]
        join_clauses = []
        for i, (tname1, _, mcols1) in enumerate(tinfos[1:], 1):
            eqs = []
            for c0, c1 in zip(listify(mcols0), listify(mcols1)):
                if c1:
                    # allows expression such as 'col + 4' for 'c1',
                    # for example. somewhat sneaky though
                    if isinstance(c1, str):
                        eqs.append(f't0.{c0} = t{i}.{c1}')
                    else:
                        # c1 comes with a binary operator like ">=", ">" ...
                        binop, c1 = c1
                        eqs.append(f't0.{c0} {binop} t{i}.{c1}')

            join_clauses.\
                append(f"left join {tname1} as t{i} on {' and '.join(eqs)}")
        jcs = ' '.join(join_clauses)

        allcols = []
        for i, (_, cols, _) in enumerate(tinfos):
            for c in listify(cols):
                if c == '*':
                    allcols += [f't{i}.{c1}'
                                for c1 in self.
                                _cols(f'select * from {tinfos[i][0]}')]
                else:
                    allcols.append(f't{i}.{c}')

        # create indices
        ind_tnames = []
        for tname, _, mcols in tinfos:
            mcols1 = list(dict.fromkeys(c if isinstance(c, str) else c[1]
                                        for c in listify(mcols) if c))
            ind_tname = tname + _random_string(10)
            # allows expression such as 'col + 4' for indexing, for example.
            # https://www.sqlite.org/expridx.html
            self._cursor.execute(f"""
            create index {ind_tname} on {tname}({', '.join(mcols1)})""")

        query = f"""
        create table {name} as select
        {', '.join(allcols)} from {tname0} as t0 {jcs}
        """
        self._cursor.execute(query)

        # drop indices, not so necessary
        for ind_tname in ind_tnames:
            self._cursor.execute(f"drop index {ind_tname}")

    def export(self, tables):
        for table in listify(tables):
            table, ext = os.path.splitext(table)
            if ext.lower() == '.xlsx':
                rs = self.fetch(f'select * from {table}')
                book = Workbook()
                sheet = book.active
                r0, rs = spy(rs)
                header = list(r0[0])
                sheet.append(header)
                for r in rs:
                    sheet.append(list(r.values()))
                book.save(os.path.join(_WS[0], f'{table}.xlsx'))

            else:
                with open(os.path.join(_WS[0], table + '.csv'), 'w',
                          encoding='utf-8', newline='') as f:
                    rs = self.fetch(f'select * from {table}')
                    r0, rs = spy(rs)
                    if r0 == []:
                        raise NoRowToWrite
                    fieldnames = list(r0[0])
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for r in rs:
                        writer.writerow(r)

    def _cols(self, query):
        return [c[0] for c in self._cursor.execute(query).description]

    def _size(self, table):
        self._cursor.execute(f"select count(*) as c from {table}")
        return self._cursor.fetchone()['c']


def get(tname, cols=None, df=False):
    """Get a list of rows (the whole table) ordered by cols.

    :param tname: table name
    :param cols: comma separated string
    :returns: a list of rows
    """
    def getit(c):
        if tname in c.get_tables():
            if cols:
                sql = f"""select * from {tname}
                          order by {','.join(listify(cols))}"""
            else:
                sql = f"select * from {tname}"
            return pd.read_sql(sql, c._conn) if df else list(c.fetch(sql))
        else:
            raise NoSuchTableFound(tname)

    tname = tname.strip()
    c = _CONN[0]
    if c and c._is_connected:
        try:
            return getit(c)
        except NoSuchTableFound as e:
            # when it's possible to execute once the other jobs are done
            if tname in _JOBS:
                raise SkipThisTurn
            else:
                raise e
    else:
        with _connect(_DBNAME) as c:
            return getit(c)


def _dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def _flatten(seq):
    # You could think of enhancing the performance here.
    # But I doubt that it's worth the trouble
    for x in seq:
        if isinstance(x, dict):
            yield x
        # ignores None
        elif x is not None:
            yield from x


def _applyfn(fn, seq):
    yield from _flatten(fn(rs) for rs in seq)


def _tqdm(seq, total, by):
    if by:
        with tqdm(seq, total=total) as pbar:
            for rs in seq:
                yield rs
                pbar.update(len(rs))
    else:
        with tqdm(seq, total=total) as pbar:
            for r in seq:
                yield r
                pbar.update(1)


def _execute(c, job):
    cmd = job['cmd']
    if cmd == 'load':
        c.load(job['file'], job['output'], delimiter=job['delimiter'],
               quotechar=job['quotechar'], encoding=job['encoding'],
               fn=job['fn'])

    elif cmd == 'apply':
        if not job['parallel']:
            _execute_serial_apply(c, job)
        else:
            _execute_parallel_apply(c, job)

    # The only place where 'insert' is not used
    elif cmd == 'join':
        c.join(job['args'], job['output'])

    elif cmd == 'mzip':
        gseqs = [groupby(c.fetch(f"""select * from {table}
                                     order by {', '.join(listify(cols))}"""),
                         _build_keyfn(cols))
                 for table, cols in job['data']]
        fn = job['fn']() if _is_thunk(job['fn']) else job['fn']
        seq = _flatten(fn(*xs) for xs in
                       tqdm(step(gseqs, stop_short=job['stop_short'])))
        c.insert(seq, job['output'])

    elif cmd == 'concat':
        def gen():
            for inp in job['inputs']:
                for r in c.fetch(f"select * from {inp}"):
                    yield r
        c.insert(tqdm(gen()), job['output'])


def _line_count(fname, encoding, newline):
    # read the number of lines fast
    def blocks(fp):
        while True:
            b = fp.read(65536)
            if not b:
                break
            yield b
    fname1 = os.path.join(_WS[0], fname)
    with open(fname1, encoding=encoding, newline=newline, errors='ignore') as f:
        # subtract -1 for a header
        return (sum(bl.count("\n") for bl in blocks(f))) - 1


def _execute_serial_apply(c, job):
    tsize = c._size(job['inputs'][0])
    logger.info(f"processing {job['cmd']}: {job['output']}")
    seq = c.fetch(f"select * from {job['inputs'][0]}", job['by'])
    evaled_fn = job['fn']() if _is_thunk(job['fn']) else job['fn']
    seq1 = _applyfn(evaled_fn, _tqdm(seq, tsize, job['by']))
    c.insert(seq1, job['output'])


# sqlite3 in osx can't handle multiple connections properly.
# Do not use multiprocessing.Queue. It's too pricy for this work.
def _execute_parallel_apply(c, job):
    max_workers = psutil.cpu_count(logical=False)
    tsize = c._size(job['inputs'][0])
    # Deal with corner cases
    if max_workers < 2 or tsize < 2:
        _execute_serial_apply(c, job)
        return

    itable = job['inputs'][0]
    tdir = os.path.join(_WS[0], _TEMP)
    if not os.path.exists(tdir):
        os.makedirs(tdir)
    evaled_fn = job['fn']() if _is_thunk(job['fn']) else job['fn']
    tcon = 'con' + _random_string(9)
    ttable = "tbl" + _random_string(9)
    breaks = [int(i * tsize / max_workers)
              for i in range(1, max_workers)]

   # perform all the process per partition.
    def _proc(dbfile, cut):
        query = f"""select * from {ttable}
                    where _ROWID_ > {cut[0]} and _ROWID_ <= {cut[1]}
                 """
        with _connect(dbfile) as c1:
            n = cut[1] - cut[0]
            seq = _applyfn(evaled_fn,
                           _tqdm(c1.fetch(query, by=job['by']),
                                 n, by=job['by']))
            try:
                c1.insert(seq, job['output'])
            except NoRowToInsert:
                pass

    def _collect_tables(dbfiles):
        succeeded_dbfiles = []
        for dbfile in dbfiles:
            with _connect(dbfile) as c1:
                if job['output'] in c1.get_tables():
                    succeeded_dbfiles.append(dbfile)

        if succeeded_dbfiles == []:
            raise NoRowToInsert

        with _connect(succeeded_dbfiles[0]) as c1:
            # query order is not actually specified
            ocols = c1._cols(f"select * from {job['output']}")
        c._cursor.execute(_create_statement(job['output'], ocols))

        # collect tables from dbfiles
        for dbfile in succeeded_dbfiles:
            c._cursor.execute(f"attach database '{dbfile}' as {tcon}")
            c._cursor.execute(f"""insert into {job['output']}
                                  select * from {tcon}.{job['output']}
                               """)
            c._conn.commit()
            c._cursor.execute(f"detach database {tcon}")

    def _delete_dbfiles(dbfiles):
        with _delayed_keyboard_interrupts():
            for dbfile in dbfiles:
                if os.path.exists(dbfile):
                    os.remove(dbfile)

    # condition for parallel work by group
    if job['by']:
        def new_breaks(breaks, group_breaks):
            index = 0
            result = []
            n = len(breaks)
            for b0 in group_breaks:
                if index >= n:
                    break
                if b0 < breaks[index]:
                    continue
                while breaks[index] <= b0:
                    index += 1
                    if index >= n:
                        break
                result.append(b0)
            return result

        try:
            dbfile0 = os.path.join(_TEMP, _random_string(10))
            c._cursor.execute(f"attach database '{dbfile0}' as {tcon}")
            c._cursor.execute(f"""create table {tcon}.{ttable} as select * from {itable}
                                  order by {','.join(job['by'])}
                               """)
            c._conn.commit()
            group_breaks = \
                [list(r.values())[0] for r in c._cursor.execute(
                    f"""select _ROWID_ from {tcon}.{ttable}
                        group by {','.join(job['by'])} having MAX(_ROWID_)
                    """)]
            if len(group_breaks) == 1:
                _execute_serial_apply(c, job)
                return
            breaks = new_breaks(breaks, group_breaks)

            dbfiles = [dbfile0] + [os.path.join(_TEMP, _random_string(10))
                                   for _ in range(len(breaks))]
            exe = Pool(len(dbfiles))

            c._cursor.execute(f"detach database {tcon}")
            logger.info(
                f"processing {job['cmd']}: {job['output']}"
                f" (multiprocessing: {len(breaks) + 1})")
            for dbfile in dbfiles[1:]:
                copyfile(dbfiles[0], dbfile)

            exe.map(_proc, dbfiles, zip([0] + breaks, breaks + [tsize]))
            _collect_tables(dbfiles)
        finally:
            _delete_dbfiles(dbfiles)

    # non group parallel work
    else:
        try:
           # remove duplicates
            breaks = list(dict.fromkeys(breaks))
            dbfiles = [os.path.join(_TEMP, _random_string(10))
                       for _ in range(len(breaks) + 1)]
            exe = Pool(len(dbfiles))
            c._cursor.execute(f"attach database '{dbfiles[0]}' as {tcon}")
            c._cursor.execute(f"""create table {tcon}.{ttable}
                                  as select * from {itable}
                               """)
            c._conn.commit()
            c._cursor.execute(f"detach database {tcon}")
            for dbfile in dbfiles[1:]:
                copyfile(dbfiles[0], dbfile)

            logger.info(
                f"processing {job['cmd']}: {job['output']}"
                f" (multiprocessing: {len(breaks) + 1})")

            exe.map(_proc, dbfiles, zip([0] + breaks, breaks + [tsize]))
            _collect_tables(dbfiles)
        finally:
            _delete_dbfiles(dbfiles)


def load(file=None, fn=None, delimiter=None, quotechar='"', encoding='utf-8'):
    """Forms load instruction."""
    return {'cmd': 'load',
            'file': file,
            'fn': fn,
            'delimiter': delimiter,
            'quotechar': quotechar,
            'encoding': encoding,
            'inputs': [],
            }


def apply(fn=None, data=None, by=None, parallel=False):
    """Forms apply instruction."""
    return {
        'cmd': 'apply',
        'fn': fn,
        'inputs': [data],
        'by': listify(by) if by else None,
        'parallel': parallel,
    }


def join(*args):
    """Forms join instruction."""
    inputs = [arg[0] for arg in args]
    return {
        'cmd': 'join',
        'inputs': inputs,
        'args': args,
    }


def mzip(fn, data=None, stop_short=False):
    """Forms mzip instruction."""
    return {
        'cmd': 'mzip',
        'fn': fn,
        'inputs': [table for table, _ in data],
        'data': data,
        'stop_short': stop_short,
    }


def concat(inputs):
    """Forms concat instruction."""
    return {
        'cmd': 'concat',
        'inputs': listify(inputs)
    }


def register(**kwargs):
    """Register processes as keyword args.

    .. highlight:: python
    .. code-block:: python

        import mapon as mo

        sp.register(
            table_name = sp.load('sample.csv'),
            table_name1 = sp.apply(simple_process, 'table_name'),
        )

        mo.run()
    """
    for k, _ in kwargs.items():
        if _JOBS.get(k, False):
            raise TableDuplication(k)
    _JOBS.update(kwargs)


def run(**kwargs):
    """Run registered processes, only those that are not in the database and\
        also that depend on those missing processes.

    :param ws: working space, the default is where the script is.
    :type ws: path as str
    :param locale: The default is en_US.UTF-8
        (English_United States.1252 for Windows)
    :param msg: You may not want visual noises in the terminal.
    :param refresh: You may want to rerun the script over and over again,
        changing the code a bit. It could be tedious to delete the table
        everytime in those circumstances.

        Pass table names in string, for example 'table1, table2'.
    :type refresh: comma separated string of table names
    :param export: Export tables in csv. for example 'table1, table2'
        Those CSVs will be in the workspace as table1.csv, table2.csv.
    :type export: comma separated string of table names
    :returns: Actually returns something for testing purposes,
        but not relavant.
    """
    global _CONFIG
    try:
        default_configs = {k: v for k, v in _CONFIG.items()}
        for k, v in kwargs.items():
            if k not in _CONFIG:
                raise UnknownConfig(k)
            _CONFIG[k] = v
        return _run()
    finally:
        _CONFIG = default_configs


def _run():
    def append_output(kwargs):
        for k, v in kwargs.items():
            v['output'] = k
        return [v for _, v in kwargs.items()]

    def find_required_tables(jobs):
        tables = set()
        for job in jobs:
            for table in job['inputs']:
                tables.add(table)
            tables.add(job['output'])
        return tables

    # depth first search
    def dfs(graph, path, paths=[]):
        datum = path[-1]
        if datum in graph:
            for val in graph[datum]:
                new_path = path + [val]
                paths = dfs(graph, new_path, paths)
        else:
            paths += [path]
        return paths

    # graph: {input: [out1, out2, ...]}
    def build_graph(jobs):
        graph = {}
        for job in jobs:
            for ip in job['inputs']:
                if graph.get(ip):
                    graph[ip].add(job['output'])
                else:
                    graph[ip] = {job['output']}
        for x in graph:
            graph[x] = list(graph[x])
        return graph

    def render_graph(graph, jobs):
        dot = Digraph()
        for k, v in graph.items():
            dot.node(k, k)
            if k != v:
                for v1 in v:
                    dot.edge(k, v1)
        for job in jobs:
            if job['cmd'] == 'load':
                dot.node(job['output'], job['output'])
        dot.render(os.path.join(_WS[0], _GRAPH_NAME))

    jobs = append_output(_JOBS)
    required_tables = find_required_tables(jobs)
    with _connect(_DBNAME) as c:
        _CONN[0] = c

        def delete_after(missing_table, paths):
            for path in paths:
                if missing_table in path:
                    for x in path[path.index(missing_table):]:
                        c.drop(x)

        def get_missing_tables():
            existing_tables = c.get_tables()
            return [table for table in required_tables
                    if table not in existing_tables]

        def find_jobs_to_do(jobs):
            missing_tables = get_missing_tables()
            result = []
            for job in jobs:
                for table in job['inputs'] + [job['output']]:
                    if table in missing_tables:
                        result.append(job)
                        break
            return result

        def is_doable(job):
            missing_tables = get_missing_tables()
            return all(table not in missing_tables for table in job['inputs'])\
                and job['output'] in missing_tables

        graph = build_graph(jobs)
        try:
            render_graph(graph, jobs)
        except Exception:
            pass

        # delete tables in 'refresh'
        if _CONFIG['refresh']:
            c.drop(listify(_CONFIG['refresh']))

        starting_points = [job['output']
                           for job in jobs if job['cmd'] == 'load']
        paths = []
        for sp in starting_points:
            paths += dfs(graph, [sp], [])

        for mt in get_missing_tables():
            delete_after(mt, paths)

        jobs_to_do = find_jobs_to_do(jobs)
        initial_jobs_to_do = list(jobs_to_do)
        logger.info(f'To Create: {[j["output"] for j in jobs_to_do]}')

        while jobs_to_do:
            cnt = 0
            for i, job in enumerate(jobs_to_do):
                if is_doable(job):
                    try:
                        if job['cmd'] != 'apply':
                            logger.info(
                                f"processing {job['cmd']}: {job['output']}")
                        _execute(c, job)

                    except SkipThisTurn:
                        continue

                    except Exception as e:

                        if isinstance(e, NoRowToInsert):
                            # Many times you want it to be silenced
                            # because you want to test it before actually
                            # write the code
                            logger.warning(
                                f"No row to insert: {job['output']}")
                        else:
                            logger.error(f"Failed: {job['output']}")
                            logger.error(f"{type(e).__name__}: {e}",
                                         exc_info=True)

                        try:
                            c.drop(job['output'])
                        except Exception:
                            pass

                        logger.warning(
                            f"Unfinished: "
                            f"{[job['output'] for job in jobs_to_do]}")
                        return (initial_jobs_to_do, jobs_to_do)
                    del jobs_to_do[i]
                    cnt += 1
            # No jobs can be done anymore
            if cnt == 0:
                for j in jobs_to_do:
                    logger.warning(f'Unfinished: {j["output"]}')
                    for t in j['inputs']:
                        if t not in c.get_tables():
                            logger.warning(f'Table not found: {t}')
                return (initial_jobs_to_do, jobs_to_do)
        # All jobs done well
        if _CONFIG['export']:
            c.export(listify(_CONFIG['export']))

        return (initial_jobs_to_do, jobs_to_do)


def _random_string(nchars):
    """Generate a random string of lengh 'n' with alphabets and digits."""
    chars = string.ascii_letters + string.digits
    return ''.join(random.SystemRandom().choice(chars)
                   for _ in range(nchars))


# primary keys are too much for non-experts
def _create_statement(name, colnames):
    """Create table if not exists foo (...).

    Note:
        Every type is numeric.
        Table name and column names are all lowercased
    """
    # every col is numeric, this may not be so elegant but simple to handle.
    # If you want to change this, Think again
    schema = ', '.join([col + ' ' + 'numeric' for col in colnames])
    return "create table if not exists %s (%s)" % (name, schema)


# column can contain spaces. So you must strip them all
def _insert_statement(name, d):
    """Generate an insert statememt.

    ex) insert into foo values (:a, :b, :c, ...)
    """
    keycols = ', '.join(":" + c.strip() for c in d)
    return "insert into %s values (%s)" % (name, keycols)


def _read_csv(filename, delimiter=',', quotechar='"',
              encoding='utf-8', newline="\n"):
    with open(os.path.join(_WS[0], filename),
              encoding=encoding, newline=newline) as f:
        header = [c.strip() for c in f.readline().split(delimiter)]
        yield from csv.DictReader(f, fieldnames=header,
                                  delimiter=delimiter, quotechar=quotechar)


def _read_sas(filename):
    filename = os.path.join(_WS[0], filename)
    with SAS7BDAT(filename) as f:
        reader = f.readlines()
        header = [c.strip() for c in next(reader)]
        for line in reader:
            yield {k: v for k, v in zip(header, line)}


def _read_df(df):
    cols = df.columns
    header = [c.strip() for c in df.columns]
    for _, r in df.iterrows():
        yield {k: v for k, v in zip(header, ((str(r[c]) for c in cols)))}


# this could be more complex but should it be?
def _read_excel(filename):
    filename = os.path.join(_WS[0], filename)
    # it's OK. Excel files are small
    df = pd.read_excel(filename, keep_default_na=False)
    yield from _read_df(df)


# raises a deprecation warning
def _read_stata(filename):
    filename = os.path.join(_WS[0], filename)
    chunk = 10_000
    for xs in pd.read_stata(filename, chunksize=chunk):
        yield from _read_df(xs)


def _is_reserved(x):
    return x.upper() in _RESERVED_KEYWORDS


def _is_thunk(fn):
    return len(signature(fn).parameters) == 0
