"""
"""
# 'connect' is deprecated,
# Often times I feel isconsec is too much
from .core import Row, Rows, readxl, process, Load, Map, Union, Join, connect, tocsv, drop
from .util import isnum, dconv, dmath, isconsec, grouper


__all__ = ['process', 'Row', 'Rows', 'isnum', 'dconv', 'dmath', 'readxl', 'grouper',
           'isconsec', 'Map', 'Load', 'Union', 'Join', 'connect', 'tocsv', 'drop']


