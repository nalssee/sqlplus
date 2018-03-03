import os
import sys
import unittest
import sqlite3
import pandas as pd

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from sqlplus import Row, Rows, connect, dmath, isnum, dconv, grouper, setdir


setdir('data')



# pns (portfolio numbering based on the first date, and assign the same for
# all the follow up rows, since you hold the portfolio)
def pns(rs, d, dcol, icol, dep=False):
    fdate = rs[0][dcol]
    rs0 = rs.where(lambda r: r[dcol] == fdate)
    rs1 = rs.where(lambda r: r[dcol] != fdate)
    rs0.numbering(d, dep)
    rs1.follow(rs0, icol, ['pn_' + x for x in list(d)])


class TestRows(unittest.TestCase):
    def test_init(self):
        seq = (Row(x=i) for i in range(3))
        rs = Rows(seq)
        self.assertEqual(len(rs.rows), 3)
        self.assertEqual(len(rs), 3)

        # No verifications

    def test_getitem(self):
        seq = (Row(x=i, y=i + 1) for i in range(3))
        rs = Rows(seq)
        self.assertEqual(rs['x'], [0, 1, 2])
        self.assertEqual(rs['x, y'], [[0, 1], [1, 2], [2, 3]])

        self.assertEqual(rs[0], rs.rows[0])
        rs1 = rs[0:2]
        rs.rows += [Row(x=10, y=11)]
        self.assertEqual(len(rs1), 2)
        # rs itself is not changed, shallow copy
        self.assertEqual(len(rs), 4)
        rs1[0].x += 1
        self.assertEqual(rs[0].x, 1)

    def test_setitem(self):
        seq = (Row(x=i, y=i + 1) for i in range(3))
        rs = Rows(seq)
        rs[1] = 30
        # non Row is assigned
        with self.assertRaises(Exception):
            rs.df()
        rs[1] = Row(x=10, y=30, z=50)
        self.assertEqual(rs[1].values, [10, 30, 50])
        with self.assertRaises(Exception):
            rs.df()
        # valid rows is asigned
        rs[1] = Row(x=10, y=30)
        self.assertEqual(rs['x'], [0, 10, 2])
        rs[1:2] = [Row(x=10, y=30)]
        self.assertEqual(rs['x'], [0, 10, 2])

        with self.assertRaises(Exception):
            rs['x'] = ['a', 'b']
        rs['x'] = ['a', 'b', 'c']

    def test_delitem(self):
        seq = (Row(x=i, y=i + 1) for i in range(3))
        rs = Rows(seq)
        del rs['x']
        self.assertEqual(rs[0].columns, ['y'])
        with self.assertRaises(Exception):
            del rs['x']
        del rs['y']
        self.assertEqual(rs[0].columns, [])

    def test_isconsec(self):
        seq = []
        for i in range(10):
            seq.append(Row(date=dmath('20010128', {'days': i}, '%Y%m%d')))
        seq = Rows(seq)
        self.assertTrue(seq.isconsec('date', '1 day', '%Y%m%d'))
        del seq[3]
        self.assertFalse(seq.isconsec('date', '1 day', '%Y%m%d'))

    def test_roll(self):
        rs1 = []
        for year in range(2001, 2011):
            rs1.append(Row(date=year))

        lengths = []
        for rs0 in Rows(rs1).roll(3, 2, 'date', True):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [3, 3, 3, 3, 2])

        lengths = []
        for rs0 in Rows(rs1).roll(3, 2, 'date'):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [3, 3, 3, 3])

        rs2 = []
        start_month = '200101'
        for i in range(36):
            rs2.append(Row(date=addm(start_month, i)))

        lengths = []
        for rs0 in Rows(rs2).where(lambda r: r.date > '200103')\
                            .roll(12, 12, 'date', lambda d: addm(d, 1), True):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [12, 12, 9])

        lengths = []
        for rs0 in Rows(rs2).where(lambda r: r.date > '200103')\
                            .roll(24, 12, 'date', lambda d: addm(d, 1), False):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [24])

        rs3 = []
        start_date = '20010101'
        for i in range(30):
            rs3.append(Row(date=dmath(start_date, {'days': i}, '%Y%m%d')))

        lengths = []
        for rs0 in Rows(rs3).roll(14, 7, 'date',
                                  lambda d: dmath(d, '1 day', '%Y%m%d'), True):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [14, 14, 14, 9, 2])

        # # should be able to handle missing dates
        rs = Rows([Row(date=addm('200101', i)) for i in range(10)])
        del rs[3]
        ls = [[int(x) for x in rs1['date']]
              for rs1 in rs.roll(5, 4, 'date', lambda d: addm(d, 1), True)]
        self.assertEqual(
            ls, [[200101, 200102, 200103, 200105],
                 [200105, 200106, 200107, 200108, 200109],
                 [200109, 200110]])

    # pns is a combination of numbering and follow
    # test numbering and follow
    def test_numbering(self):
        with connect('sample.db') as c:
            # now you need yyyy column
            c.register(lambda d: dconv(d, '%Y-%m-%d', '%Y'), 'yearfn')
            c.create('select *, yearfn(date) as yyyy from acc1', 'tmpacc1')

            # oneway sorting
            c.drop('tmpacc2')
            for rs in c.fetch('tmpacc1', where='isnum(asset)',
                              roll=(3, 3, 'yyyy', True)):
                pns(rs, {'asset': 10}, dcol='yyyy', icol='id')
                c.insert(rs.isnum('pn_asset'), 'tmpacc2')

            for rs in c.fetch('tmpacc2', roll=(3, 3, 'yyyy', True)):
                xs = [len(x) for x in rs.group('yyyy')]
                # the first one must have the largest number of items
                self.assertEqual(max(xs), xs[0])

            # average them
            c.drop('tmpaccavg')
            for rs in c.fetch('tmpacc2', group='yyyy, pn_asset'):
                r = Row()
                r.date = rs[0].yyyy
                r.pn_asset = rs[0].pn_asset
                r.avgasset = rs.avg('asset')
                c.insert(r, 'tmpaccavg')

            # tests if pn numbering is correct!!
            for rs in c.fetch('tmpaccavg', roll=(3, 3, 'date', True)):
                fdate = rs[0]['date']
                rs1 = rs.where(lambda r: r.date==fdate)
                xs1 = rs1.order('pn_asset')['avgasset']
                xs2 = rs1.order('avgasset')['avgasset']
                self.assertEqual(xs1, xs2)

            c.drop('tmpacc1, tmpacc2, tmpaccavg')

    def test_numbering1(self):
        def fn1(rs):
            rs0 = rs.where(lambda r: r.a == 0)
            rs1 = rs.where(lambda r: r.a > 0)
            yield rs0
            yield rs1

        rs0 = Rows(Row(a=0) for _ in range(5))
        for i, r in enumerate(rs0, 1):
            r.b = i

        rs1 = Rows(Row(a=1) for _ in range(10))
        for i, r in enumerate(rs1, 6):
            r.b = i

        rs = rs0 + rs1

        rs['pn_a'] = ''
        rs['pn_b'] = ''
        rs.numbering({'a': fn1, 'b': 2}, dep=True)
        self.assertEqual(rs['pn_a, pn_b'],
                         [[1, 1], [1, 1],
                          [1, 2], [1, 2], [1, 2],
                          [2, 1], [2, 1], [2, 1], [2, 1], [2, 1],
                          [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])

        rs['pn_a'] = ''
        rs['pn_b'] = ''
        rs.numbering({'b': 2, 'a': fn1}, dep=True)
        self.assertEqual(rs['pn_a, pn_b'],
                         [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                          [2, 1], [2, 1], [2, 2], [2, 2], [2, 2],
                          [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])

        rs['pn_a'] = ''
        rs['pn_b'] = ''
        rs.numbering({'a': fn1, 'b': 2})
        self.assertEqual(rs['pn_a, pn_b'],
                         [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                          [2, 1], [2, 1], [2, 2], [2, 2], [2, 2],
                          [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])

        rs['pn_b'] = ''
        rs['pn_a'] = ''
        rs.numbering({'b': 2, 'a': fn1})
        self.assertEqual(rs['pn_a, pn_b'],
                         [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1],
                          [2, 1], [2, 1], [2, 2], [2, 2], [2, 2],
                          [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]])

    def test_numbering2d(self):
        with connect('sample.db') as c:
            # now you need yyyy column
            c.register(lambda d: dconv(d, '%Y-%m-%d', '%Y'), 'yearfn')
            c.create('select *, yearfn(date) as yyyy from acc1', 'tmpacc1')

            c.drop('tmpacc2')
            for rs in c.fetch('tmpacc1', where='isnum(asset)',
                              roll=(8, 8, 'yyyy', True)):
                pns(rs, {'asset': 4, 'ppe': 4}, dcol='yyyy', icol='id')
                c.insert(rs.isnum('pn_asset, pn_ppe'), 'tmpacc2')

            import statistics as st
            for rs in c.fetch('tmpacc2', where='yyyy >= 1988', group='yyyy'):
                for i in range(1, 5):
                    xs = []
                    for j in range(1, 5):
                        n = len(rs.where(lambda r: r.pn_asset==i and r.pn_ppe==j))
                        xs.append(n)
                    self.assertTrue(st.stdev(xs) >= 12)

            # dependent sort
            c.drop('tmpacc2')
            for rs in c.fetch('tmpacc1', where='isnum(asset)',
                              roll=(8, 8, 'yyyy', True)):
                pns(rs, {'asset': 4, 'ppe': 4},
                    dcol='yyyy', icol='id', dep=True)
                c.insert(rs.isnum('pn_asset, pn_ppe'), 'tmpacc2')

            for rs in c.fetch('tmpacc2', where='yyyy >= 1988', group='yyyy'):
                for i in range(1, 5):
                    xs = []
                    for j in range(1, 5):
                        n = len(rs.where(lambda r: r.pn_asset==i and r.pn_ppe==j))
                        xs.append(n)
                    # number of items ought to be about the same
                    # Test not so sophisticated
                    self.assertTrue(st.stdev(xs) < 12)


# This should be defined in 'main' if you want to exploit multiple cores
# in Windows, The function itself here is just a giberrish for testing

class TestSQLPlus(unittest.TestCase):
    # apply is removed but the following works
    def test_apply(self):
        def to_month(r):
            r.date = dconv(r.orderdate, '%Y-%m-%d', '%Y%m')
            return r

        with connect('sample.db') as q:
            tseq = (to_month(r) for r in q.fetch('orders'))
            q.insert(tseq, 'orders1')

            ls = []
            for rs in q.fetch('orders1', group='date'):
                ls.append(len(rs))

            self.assertEqual(ls, [22, 25, 23, 26, 25, 31, 33, 11])
            self.assertEqual(len(q.rows('orders1')),
                             sum([22, 25, 23, 26, 25, 31, 33, 11]))

            ls = []
            for rs in q.fetch('orders1', roll=(3, 2, 'date', True)):
                for rs1 in rs.group('shipperid'):
                    ls.append(len(rs1))
            self.assertEqual([sum(ls1) for ls1 in grouper(ls, 3)],
                             [70, 74, 89, 44])
            q.drop('orders1')

    def test_to_csv(self):
        with connect('sample.db') as c:
            c.to_csv('categories', 'foo.csv')
            a = c.rows('categories')
            c.drop('foo')
            c.load('foo.csv')
            b = c.rows('foo')
            for a1, b1 in zip(a, b):
                self.assertEqual(a1.values, b1.values)
            os.remove(os.path.join('data', 'foo.csv'))

    def test_insert(self):
        with connect('sample.db') as c:
            c.drop('foo')
            for rs in c.fetch('orders',  group='shipperid'):
                r = rs[0]
                r.n = len(rs)
                c.insert(r, 'foo')
            rs = c.rows('foo')
            self.assertEqual(rs['n'], [54, 74, 68])

            # the following must not raise exceptions
            c.insert(Rows([]), 'foo')

            c.drop('foo')

            def foo():
                for i in range(10):
                    xs = []
                    for j in range(3):
                        xs.append(Row(x=j))
                    yield Rows(xs)
            c.insert(foo(), 'foo')
            self.assertEqual(len(c.rows('foo')), 30)

            c.drop('foo')

            def foo():
                for i in range(10):
                    xs = []
                    for j in range(3):
                        xs.append(Row(x=j))
                    yield xs
            c.insert(foo(), 'foo')
            self.assertEqual(c.rows('foo')[:6]['x'], [0, 1, 2, 0, 1, 2])


# for pmap, this fn must be in top level, in Windwos machine.
def fn(r, a):
    r.x = a
    return r


class TestMisc(unittest.TestCase):
    def test_pmap(self):
        with connect(':memory:') as c:
            rs = [Row(a=i) for i in range(10)]
            rs1 = [Row(a=i) for i in range(10)]
            for i in range(4):
                # it takes much much longer!!
                for r in pmap(fn, rs, args=(i,),  max_workers=2):
                    c.insert(r, 'foo')
                # single core version
                for r in pmap(fn, rs1, args=(i,),  max_workers=1):
                    c.insert(r, 'bar')
            xs = list(c.fetch('foo', group='x'))
            for i, x in enumerate(xs):
                self.assertEqual(x[0].x, i)
                self.assertEqual(len(x), 10)
            for a, b in zip(c.rows('foo'), c.rows('bar')):
                self.assertEqual(a.values, b.values)


if __name__ == "__main__":
    ws_path = os.path.join(os.getcwd(), '')
    fname = os.path.join(ws_path, 'sample.db')
    if os.path.isfile(fname):
        os.remove(fname)
    # First load csv files in workspace to sqlite db
    with connect('sample.db') as c:
        for f in os.listdir(ws_path):
            if f.endswith('.csv'):
                c.load(f)
    unittest.main()