
import os
import sys

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from sqlplus.core import *
from sqlplus.util import isnum,  \
    prepend_header, pmap, grouper, same, ymd, read_date, listify

from datetime import * 
from dateutil.relativedelta import *

print(ymd('3 years', '%Y')(2003))
date = 2013
fmt = "%Y"
unit = 'years'
n = 30

print(datetime.strptime(str(date), fmt) + relativedelta(years=30))
from itertools import groupby
print(next(x for _, x in groupby(range(10), key=lambda x: [])))