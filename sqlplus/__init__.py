"""
"""
# reexport some date utils
from dateutil.relativedelta import relativedelta as rd
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
from datetime import datetime

from .core import Row, Rows, dbopen, setwd, getwd, SQLPlus
from .util import pmap, isnum, isconsec, grouper

strptime = datetime.strptime

__all__ = ['dbopen', 'SQLPlus', 'Row', 'Rows', 'setwd', 'getwd',
           'isnum', 'isconsec', 'grouper', 'pmap',
           'strptime', 'parse', 'relativedelta', 'rd']
