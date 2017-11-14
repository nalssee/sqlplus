import os
import sys
import unittest

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from sqlplus.core import Row, Rows, dbopen
from sqlplus.util import ymd, isnum, read_date, grouper, breakpoints, pmap
from sqlplus.load import read_csv
from sqlplus import core


class TestRow(unittest.TestCase):
    def test_init(self):
        r = Row()
        self.assertEqual(r.columns, [])
        self.assertEqual(r.values, [])

        r = Row(a=10, b=20)
        self.assertEqual(r.columns, ['a', 'b'])
        self.assertEqual(r.values, [10, 20])

        r = Row(b=10, a=20)
        self.assertEqual(r.columns, ['b', 'a'])

    def test_copy(self):
        r1 = Row(a=10, b='test')
        r2 = r1.copy()

        self.assertEqual(r1.columns, r2.columns)
        self.assertEqual(r1.values, r2.values)

    def test_getattr(self):
        r1 = Row(a=10, b='test')
        self.assertEqual(r1.a, 10)
        self.assertEqual(r1['b'], 'test')

    def test_setattr(self):
        r1 = Row(a=10, b='test')
        r1.a = 20
        self.assertEqual(r1.a, 20)
        r1['b'] = 'test1'
        self.assertEqual(r1.b, 'test1')
        r1.c = [1, 2, 3]
        self.assertEqual(r1.c, [1, 2, 3])

    def test_delattr(self):
        r1 = Row(a=10, b='test')
        del r1.a
        self.assertEqual(r1.columns, ['b'])

        with self.assertRaises(Exception):
            del r1.a

        del r1['b']
        self.assertEqual(r1.columns, [])


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
            rs.show()
        rs[1] = Row(x=10, y=30, z=50)
        self.assertEqual(rs[1].values, [10, 30, 50])
        with self.assertRaises(Exception):
            rs.show()
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

    def test_lzip(self):
        rs1 = Rows(Row(year=2001 + i) for i in range(10))
        del rs1[7]
        rs2 = Rows(Row(year=2001 + i) for i in range(10))
        rs2 = rs2[1:-1]
        del rs2[3]
        rs3 = Rows(Row(year=2001 + i) for i in range(10))
        rs3.rows.append(Row(year=2013))
        rs3 = rs3[3:]

        ns = []
        for x in rs1.lzip('year', rs2, rs3):
            ns.append(len([x1 for x1 in x if x1 is None]))
        self.assertEqual(ns, [2, 1, 1, 0, 1, 0, 0, 0, 1])

    def test_roll(self):
        rs1 = []
        for year in range(2001, 2011):
            rs1.append(Row(date=str(year)))

        lengths = []
        for rs0 in Rows(rs1).roll(3, 2, 'date', ymd('1 year', '%Y')):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [3, 3, 3, 3, 2])

        rs2 = []
        start_month = '200101'
        for i in range(36):
            rs2.append(Row(date=ymd(f'{i} months', '%Y%m')(start_month)))

        lengths = []
        for rs0 in Rows(rs2).where('date > "200103"')\
                            .roll(12, 12, 'date', ymd('1 month', '%Y%m')):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [12, 12, 9])

        lengths = []
        for rs0 in Rows(rs2).where("date > '200103'")\
                            .roll(24, 12, 'date', ymd('1 month', '%Y%m')):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [24, 21, 9])

        rs3 = []
        start_date = '20010101'
        for i in range(30):
            rs3.append(Row(date=ymd(f'{i} days', '%Y%m%d')(start_date)))

        lengths = []
        for rs0 in Rows(rs3).roll(14, 7, 'date', ymd('1 day', '%Y%m%d')):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [14, 14, 14, 9, 2])

        # should be able to handle missing dates
        rs = Rows([Row(date=ymd(f'{i} months', '%Y%m')('200101'))
                   for i in range(10)])
        del rs[3]
        ls = [[int(x) for x in rs1['date']]
              for rs1 in rs.roll(5, 4, 'date', ymd('1 month', '%Y%m'))]
        self.assertEqual(
            ls, [[200101, 200102, 200103, 200105],
                 [200105, 200106, 200107, 200108, 200109],
                 [200109, 200110]])

    def test_order(self):
        with dbopen('sample.db') as q:
            seq = (rs[0] for rs in q.read('customers', group='country'))
            q.write(seq, 'c1')
            countries = q.rows('c1').order('country', reverse=True)['country']
            self.assertEqual(len(countries), 21)
            self.assertEqual(countries[:3], ['Venezuela', 'USA', 'UK'])
            q.drop('c1')

    def test_copy(self):
        with dbopen('sample.db') as q:
            rs = q.rows('customers')
            rs1 = rs.copy()
            rs1 = rs1[:20]
            # size not changed
            self.assertEqual(len(rs), 91)

    def test_where(self):
        with dbopen('sample.db') as q:
            rs = q.rows('customers')
            self.assertEqual(len(rs.where("""
            country="USA" and postalcode < 90000""")), 4)
            self.assertEqual(len(rs.where(lambda r: isnum(r.postalcode))), 66)

    def test_corr(self):
        with dbopen('sample.db') as q:
            rs = q.rows('customers', where='isnum(postalcode)')
            xs = rs.corr(cols='customerid, postalcode')
            a = xs['customerid'][1]
            self.assertEqual(round(a * 100), 25)
            b = xs['postalcode'][0]
            self.assertEqual(round(b * 100), 21)

    def test_isnum(self):
        with dbopen('sample.db') as q:
            rs1 = q.rows('customers', where='isnum(postalcode)')
            rs2 = q.rows('customers').isnum('postalcode')
            self.assertEqual(len(rs1), len(rs2))

    def test_avg(self):
        with dbopen('sample.db') as q:
            rs1 = q.rows('products')
            self.assertEqual(round(rs1.avg('price') * 100), 2887)
            self.assertEqual(round(rs1.avg('price', 'categoryid') * 100), 2811)

    def test_ols(self):
        with dbopen('sample.db') as q:
            rs = q.rows('products')
            res = rs.ols('price ~ supplierid + categoryid')
            self.assertEqual(len(res), 3)
            self.assertEqual(res[0].columns,
                             ['param', 'coef', 'stderr', 'tval', 'pval'])

    # def test_plot(self):
    #     with dbopen('sample.db') as q:
    #         q.rows('orderdetails').plot()

    def test_truncate(self):
        with dbopen('sample.db') as q:
            rs = q.rows('products')
            self.assertEqual(len(rs.truncate('price', 0.1)), 61)

    def test_winsorize(self):
        with dbopen('sample.db') as q:
            rs = q.rows('products')
            self.assertEqual(round(rs.avg('price') * 100), 2887)
            rs = rs.winsorize('price', 0.2)
            self.assertEqual(round(rs.avg('price') * 100), 2296)

    def test_group(self):
        with dbopen('sample.db') as q:
            rs = q.rows('customers')
            ls = []
            for rs1 in rs.group('country'):
                ls.append(len(rs1))
            self.assertEqual(ls, [3, 2, 2, 9, 3, 2, 2, 11, 11,
                                  1, 3, 5, 1, 1, 2, 5, 2, 2, 7, 13, 4])

    def test_show(self):
        with dbopen('sample.db') as q:
            rs = q.rows('customers')
            rs.show(file='sample.csv')
            with dbopen(':memory:') as q1:
                q1.write(rs, 'foo')
                self.assertEqual(len(q1.rows('foo')), 91)
                os.remove(os.path.join(core.WORKSPACE, 'sample.csv'))

    def test_df(self):
        with dbopen('sample.db') as q:
            rs = q.rows('customers')
            self.assertEqual(rs.df().shape, (91, 7))

    def test_pn(self):
        # No need to order first
        with dbopen('sample.db') as q:
            rs = q.rows('orderdetails')
            bps = breakpoints(rs['quantity'], [0.3, 0.7])
            rs.pn('quantity', bps)
            self.assertEqual(len(rs.where('pn_quantity=1')), 132)
            self.assertEqual(len(rs.where('pn_quantity=2')), 214)
            self.assertEqual(len(rs.where('pn_quantity=3')), 172)


# This should be defined in 'main' if you want to exploit multiple cores
# in Windows, The function itself here is just a giberrish for testing
def avg_id(rs):
    r = Row(date=read_date(rs[0].orderdate, '%Y-%m-%d', '%Y%m'))
    r.orderid = round(rs.avg('orderid'))
    r.customerid = round(rs.avg('orderid'))
    r.employeeid = round(rs.avg('employeeid'))
    r.shipperid = rs[0].shipperid
    return r


class TestSQLPlus(unittest.TestCase):
    def test_apply(self):
        def to_month(r):
            r.date = read_date(r.orderdate, '%Y-%m-%d', '%Y%m')
            return r

        with dbopen('sample.db') as q:
            tseq = (to_month(r) for r in q.read('orders'))
            q.write(tseq, 'orders1')

            ls = []
            for rs in q.read('orders1', group='date'):
                ls.append(len(rs))

            self.assertEqual(ls, [22, 25, 23, 26, 25, 31, 33, 11])
            self.assertEqual(len(q.rows('orders1')),
                             sum([22, 25, 23, 26, 25, 31, 33, 11]))

            ls = []
            for rs in q.read('orders1',
                             roll=(3, 2, 'date', ymd('1 month', '%Y%m'))):
                ls.append(len(rs))
            self.assertEqual(ls, [70, 74, 89, 44])

            ls = []
            for rs in q.read('orders1', group='shipperid',
                             roll=(3, 2, 'date', ymd('1 month', '%Y%m'))):
                ls.append(len(rs))
            self.assertEqual([sum(ls1) for ls1 in grouper(ls, 3)],
                             [70, 74, 89, 44])
            q.drop('orders1')

    def test_register(self):
        with dbopen(':memory:') as c:
            def foo(x, y):
                return x + y

            def bar(*args):
                return sum(args)

            def foo1(rs):
                sum = 0
                for r in rs:
                    sum += r.a * r.b
                return sum

            def bar1(rs):
                sum = 0
                for r in rs:
                    sum += r[0] * r[1]
                return sum

            c.register(foo)
            c.register(bar)
            c.registerAgg(foo1, 'a, b')
            c.registerAgg(bar1)

            c.sql("create table test(i, j, x)")
            c.sql("insert into test values (1, 3,'a')")
            c.sql("insert into test values (21, 2, 'b')")
            c.sql("insert into test values (5,3, 'a')")
            c.sql("insert into test values (20,4, 'a')")
            c.sql("insert into test values (20,'x', 'c')")
            c.sql("insert into test values (20,-1.2, 'd')")

            c.new("select foo(i, j) as val1, bar(i, j) as val2 from test",
                  'test1')
            self.assertEqual(c.rows('test1')['val1'], [4, 23, 8, 24, '', 18.8])
            self.assertEqual(c.rows('test1')['val2'], [4, 23, 8, 24, '', 18.8])

            c.new("""
            select foo1(i, j) as val1, bar1(i, j) as val2 from test group by x
            """, 'test2')
            self.assertEqual(c.rows('test2')['val1'], [98, 42, '', -24.0])
            self.assertEqual(c.rows('test2')['val2'], [98, 42, '', -24.0])

    def test_join(self):
        with dbopen('sample.db') as q:

            q.join(
                ['customers', 'customername', 'customerid'],
                ['orders', 'orderid', 'customerid'],
                name='customers1'
            )
            rs = q.rows('customers1')
            self.assertEqual(len(rs), 213)
            self.assertEqual(len(rs.isnum('orderid')), 196)
            q.drop('customers1')

            def to_month(r):
                r.date = read_date(r.orderdate, '%Y-%m-%d', '%Y%m')
                return r
            tseq = (to_month(r) for r in q.read('orders'))
            q.write(tseq, 'orders1')
            # There's no benefits in using multiple cores
            # You should know what you are doing.

            tseq = pmap(avg_id, q.read('orders1', group='date'), max_workers=2)
            q.write(tseq, 'orders2')

            # testing reel
            ls = []
            for rs in q.read('orders2',
                             roll=(5, 2, 'date', ymd('1 month', '%Y%m'))):
                ls.append(len(rs))
            self.assertEqual(ls, [5, 5, 4, 2])

            self.assertEqual(len(q.rows('orders1')), 196)

            tseq = (rs[0] for rs in q.read('orders1',
                                           group='date, customerid'))

            q.write(tseq, 'orders1', pkeys='date, customerid')
            self.assertEqual(len(q.rows('orders1')), 161)

            q.join(
                ['orders1', 'date, customerid, orderid', 'date, customerid'],
                ['orders1', 'orderid as orderid1',
                 {'date': ymd('1 month', '%Y%m'), 'customerid': None}],
                ['orders1', 'orderid as orderid2',
                 {'date': ymd('2 months', '%Y%m'), 'customerid': None}],
                ['orders1', 'orderid as orderid3',
                 {'date': ymd('3 months', '%Y%m'), 'customerid': None}],
                name='orders3'
            )

            def addmonth(x, n):
                return ymd(f'{n} months', '%Y%m')(x)

            q.register(addmonth)

            q.new("""
            select a.date, a.customerid, a.orderid,
            b.orderid as orderid1,
            c.orderid as orderid2,
            d.orderid as orderid3

            from orders1 as a

            left join orders1 as b
            on a.date = addmonth(b.date, 1) and a.customerid = b.customerid

            left join orders1 as c
            on a.date = addmonth(c.date, 2) and a.customerid = c.customerid

            left join orders1 as d
            on a.date = addmonth(d.date, 3) and a.customerid = d.customerid
            """, name='orders4')

            rs3 = q.rows('orders3')
            rs4 = q.rows('orders4')

            # primary keys check
            for r in q.sql('pragma table_info(orders3)'):
                if r[1] == 'date':
                    self.assertEqual(r[5], 1)
                elif r[1] == 'customerid':
                    self.assertEqual(r[5], 2)
                else:
                    self.assertEqual(r[5], 0)

            for r3, r4 in zip(rs3, rs4):
                self.assertEqual(r3.values, r4.values)

            q.drop('orders1, orders2, orders3, orders4')


if __name__ == "__main__":
    ws_path = os.path.join(os.getcwd(), 'workspace')
    if not os.path.isfile(os.path.join(ws_path, 'sample.db')):
        # First write csv files in workspace to sqlite db
        with dbopen('sample.db') as q:
            for f in os.listdir(ws_path):
                if f.endswith('.csv') and f != 'acc1.csv':
                    q.write(read_csv(f), f[:-4])

    unittest.main()