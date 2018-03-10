"""
"""
# 'connect' is deprecated,
# Often times I feel isconsec is too much
from .core import Row, Rows, readxl, process, Load, Map, Union, Join, \
    tocsv, drop, rename
from .util import isnum, dmath, grouper, perr


__all__ = ['process', 'Row', 'Rows', 'isnum', 'dmath', 'perr', 'readxl', 'grouper',
           'Map', 'Load', 'Union', 'Join', 'rename', 'tocsv', 'drop']


