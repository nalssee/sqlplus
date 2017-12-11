import locale
import os
import csv
import pandas as pd
import fileinput

from itertools import groupby

import sqlplus
from sqlplus.core import Row
from sqlplus.util import grouper, listify
from sas7bdat import SAS7BDAT


if os.name == 'nt':
    locale.setlocale(locale.LC_ALL, 'english_USA')
elif os.name == 'posix':
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def read_csv(filename, encoding='utf-8'):
    "Loads well-formed csv file, 1 header line and the rest is data "
    def is_empty_line(line):
        """Tests if a list of strings is empty for example ["", ""] or []
        """
        return [x for x in line if x.strip() != ""] == []

    with open(os.path.join(sqlplus.core.WORKSPACE, filename),
              encoding=encoding) as fin:
        first_line = fin.readline()[:-1]
        columns = listify(first_line)
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
            row1 = Row()
            for col, val in zip(columns, line):
                row1[col] = val
            yield row1


def read_sas(filename):
    filename = os.path.join(sqlplus.core.WORKSPACE, filename)
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
def read_excel(filename):
    filename = os.path.join(sqlplus.core.WORKSPACE, filename)
    # it's OK. Excel files are small
    df = pd.read_excel(filename)
    yield from read_df(df)


# Might be exported later.
def read_df(df):
    cols = df.columns
    for i, r in df.iterrows():
        r0 = Row()
        for c, v in zip(cols, (r[c] for c in cols)):
            r0[c] = str(v)
        yield r0


# wierd form excel file
def read_fnguide(filename, cols):
    def convert_string(x):
        # no comma
        if x.find(',') == -1:
            return x
        try:
            return locale.atoi(x)
        except ValueError:
            try:
                return locale.atof(x)
            except Exception:
                return x

    # Get firm codes
    # (number of columns(items), list of firm codes)
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

    filename = os.path.join(sqlplus.core.WORKSPACE, filename)
    with open(filename, encoding='cp949') as f:
        # throw away 8 lines
        for _ in range(8):
            f.readline()
        reader = csv.reader(f)
        # checks if the number of 'cols' corresponds with the file
        line1 = next(reader)
        n, ids = extract_ids(line1)
        # throw away 5 more lines
        for _ in range(5):
            next(reader)

        cols = listify(cols)
        assert len(cols) == n, f"{n} cols required, given {cols}"
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


def prepend_header(filename, header=None, drop=0):
    """Drop n lines and prepend header

    Args:
        filename (str)
        header (str)
        drop (int)
    """
    filename = os.path.join(sqlplus.core.WORKSPACE, filename)
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

