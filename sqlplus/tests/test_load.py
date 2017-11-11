import os
import sys
import unittest

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from sqlplus.core import dbopen
from sqlplus.load import read_csv, read_excel, read_sas, read_fnguide
from sqlplus.util import peek_first

class TestLoading(unittest.TestCase):
    def test_load_csv(self):
        with dbopen('sample1.db') as q:
            q.write(read_csv('orders.csv'), 'orders')

            self.assertEqual(len(q.rows('orders')), 196)

    def test_load_excel(self):
        with dbopen('sample1.db') as q:
            q.write(read_excel('orders.xlsx'), 'orders')
            # You may see some surprises because
            # read_excel uses pandas way of reading excel files
            # q.rows('orders1').show()
            self.assertEqual(len(q.rows('orders')), 196)

    def test_sas(self):
        with dbopen('sample1.db') as q:
            q.write(read_sas('ff5_ew_mine.sas7bdat'), 'ff5ew')
            self.assertEqual(len(q.rows('ff5ew')), 253)

    def test_fnguide(self):
        with dbopen('sample1.db') as q:
            with self.assertRaises(Exception):
                q.write(read_fnguide('acc1.csv', 'a,b,c,d'), 'acc1')

            q.write(read_fnguide('acc1.csv', 'a, b, c, d, e'), 'acc1')
            self.assertEqual(len(q.rows('acc1')), 109554)

    # You may want to test 'read_df' as well

if __name__ == "__main__":
    try:
        unittest.main()
    finally:
        os.remove(os.path.join(TESTPATH, 'workspace', 'sample1.db'))
