"""
"""
# 'connect' is deprecated,
# Often times I feel isconsec is too much
from .core import Row, Rows, readxl, process, Load, Map, Union, Join, \
    tocsv, drop, rename, describe 
from .util import isnum, dmath, grouper


__all__ = ['process', 'Row', 'Rows', 'isnum', 'dmath', 'readxl', 'grouper',
           'Map', 'Load', 'Union', 'Join', 
           'rename', 'tocsv', 'drop', 'describe']


