import os
import sys
import unittest
from itertools import chain

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from sqlplus import *
from sqlplus.core import connect


def pred1(r):
    return r.CategoryID < 5


def pred2(r):
    return r.CategoryID >= 5


def price_sum(dbname, where):
    with connect(dbname) as c:
        def fn():
            for rs in c.fetch('products', group='CategoryID', where=where):
                r = Row()
                r.CategoryID = rs[0].CategoryID
                r.psum = sum(rs['Price'])
                yield r
        c.insert(fn(), 'psum')


def overlap(xs, size, step=1):
    result = []
    for i in range(0, len(xs), step):
        result.append(Rows(chain(*xs[i: i + size])))
    return result


def month(r):
    r.yyyymm = dmath(r.orderdate, "%Y-%m-%d", "%Y-%m")
    yield r


def addmonth(date, n):
    return dmath(date, "%Y-%m", months=n)


def cnt(rs, n):
    if addmonth(rs[0].yyyymm, n - 1) == rs[-1].yyyymm:
        r = Row()
        r.yyyymm = rs[-1].yyyymm
        r.cnt = len(rs)
        r.n = n
        yield r


def allrows(c, tname):
    return Rows(c.fetch(tname))


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

    def test_avg(self):
        with connect(':memory:') as q:
            q.load('products.csv')
            rs1 = q.rows('products')
            self.assertEqual(round(rs1.avg('Price') * 100), 2887)
            self.assertEqual(round(rs1.avg('Price', 'CategoryID') * 100), 2811)

    def test_ols(self):
        with connect(':memory:') as q:
            q.load('products.csv')
            rs = q.rows('products')
            res = rs.ols('Price ~ SupplierID + CategoryID')
            self.assertEqual(len(res.params), 3)
            self.assertEqual(len(res.resid), len(rs))

    def test_truncate(self):
        with connect(':memory:') as q:
            q.load('products.csv')
            rs = q.rows('products')
            self.assertEqual(len(rs.truncate('Price', 0.1)), 61)

    def test_winsorize(self):
        with connect(':memory:') as q:
            q.load('products.csv')
            rs = q.rows('products')
            self.assertEqual(round(rs.avg('Price') * 100), 2887)
            rs = rs.winsorize('Price', 0.2)
            self.assertEqual(round(rs.avg('Price') * 100), 2296)

    def test_group(self):
        with connect(':memory:') as q:
            q.load('customers.csv')
            rs = q.rows('customers')
            ls = [len(rs1) for rs1 in rs.group('Country')]
            self.assertEqual(ls, [3, 2, 2, 9, 3, 2, 2, 11, 11,
                                  1, 3, 5, 1, 1, 2, 5, 2, 2, 7, 13, 4])

    def test_chunks(self):
        rs = Rows(Row(x=i) for i in range(10))
        for rs1 in rs.chunk(2):
            self.assertEqual(len(rs1), 5)
        # even when there are not enough elements
        ls = rs.chunk(15)
        self.assertEqual(len(ls), 15)

        with connect(':memory:') as q:
            q.load('orderdetails.csv')
            rs = q.rows('orderdetails')
            a, b, c = rs.chunk([0.3, 0.4, 0.3])
            n = len(rs)
            self.assertEqual(len(a), int(n * 0.3))
            self.assertEqual(len(b), int(n * 0.4))
            # roughly the same
            self.assertEqual(len(c), int(n * 0.3) + 1)

        rs = Rows(Row(a=i) for i in [1, 7, 3, 7])
        xs = [x['a'] for x in rs.chunk([2, 5], 'a')]
        self.assertEqual(xs, [[1], [3], [7, 7]])

        xs = [x['a'] for x in rs.chunk([2, 2.5], 'a')]
        self.assertEqual(xs, [[1], [], [3, 7, 7]])

        xs = [x['a'] for x in rs.chunk([1, 3, 5], 'a')]
        self.assertEqual(xs, [[], [1], [3], [7, 7]])

    def test_df(self):
        with connect(':memory:') as q:
            q.load('customers.csv')
            rs = q.rows('customers')
            self.assertEqual(rs.df().shape, (91, 7))

    def test_order(self):
        with connect(':memory:') as q:
            q.load('customers.csv')
            seq = (rs[0] for rs in q.fetch('customers', group='Country'))
            q.insert(seq, 'c1')
            countries = q.rows('c1').order('Country', reverse=True)['Country']
            self.assertEqual(len(countries), 21)
            self.assertEqual(countries[:3], ['Venezuela', 'USA', 'UK'])

    def test_where(self):
        with connect(':memory:') as q:
            q.load('customers.csv')
            rs = q.rows('customers')
            self.assertEqual(len(rs.where(lambda r: r.Country == "USA"
                             and r.PostalCode < 90000)), 4)
            self.assertEqual(len(rs.where(lambda r: isnum(r.PostalCode))), 66)

    def test_rows_group_and_overlap(self):
        with connect(':memory:') as c:
            c.load('products.csv')
            rss = c.rows('products').group('CategoryID')
            self.assertEqual([len(rs) for rs in rss],
                             [12, 12, 13, 10, 7, 6, 5, 12])

            sizes = []
            for x in overlap(rss, 5, 2):
                sizes.append(len(x))
            self.assertEqual(sizes, [54, 41, 30, 17])


class TestConnection(unittest.TestCase):
    def test_avg_by_group(self):
        with connect('test.db') as c:
            c.load('products.csv')

            def products_avg():
                for rs in c.fetch('products', group="CategoryID"):
                    r = Row()
                    r.CategoryID = rs[0].CategoryID
                    r.agg_price = sum(r.Price for r in rs)
                    r.n = len(rs)
                    yield r
            c.insert(products_avg(), 'products_avg')

            self.assertEqual(c.rows('products_avg')['agg_price'],
                             [455.75, 276.75, 327.08, 287.3,
                              141.75, 324.04, 161.85, 248.19])
            self.assertEqual(c.rows('products_avg')['n'],
                             [12, 12, 13, 10, 7, 6, 5, 12])

        os.remove('test.db')

    def test_group_and_overlap(self):
        with connect(':memory:') as c:
            c.load('orders.csv')
            c.create("""select *, substr(orderdate, 1, 7) as ym
            from orders""")
            ls = []
            for rs in c.fetch('orders', group='ym', overlap=(5, 2)):
                ls.append([len(rs1) for rs1 in rs.group('ym')])
            self.assertEqual(ls, [
                [22, 25, 23, 26, 25],
                [23, 26, 25, 31, 33],
                [25, 31, 33, 11],
                [33, 11]
            ])

            # simple overlap
            xs = []
            for rs in c.fetch('orders', overlap=[10, 3]):
                if len(rs) < 10:
                    xs.append(len(rs))
            self.assertEqual(xs, [7, 4, 1])

    def test_overlap_elts_modification(self):
        with connect(':memory:') as c:
            c.load('products.csv')
            x = []
            for rs in c.fetch('products', group='CategoryID', overlap=3):
                x.append(rs[0].ProductName)
                rs.set('ProductName', 0)
            # !!!!! overlap shares rows in between iterations
            self.assertEqual(x, ['Chais', 0, 0, 0, 0, 0, 0, 0])

    def test_tocsv(self):
        with connect(':memory:') as c:
            c.load('orders.csv')
            c.tocsv('orders', 'orders1.csv')
            c.drop('orders')
            c.load('orders1.csv')
            self.assertEqual(len(c.rows('orders1')), 196)
        os.remove('orders1.csv')

    def test_join(self):
        with connect(':memory:') as c:
            c.load('customers.csv')
            c.load('orders.csv')
            c.join(
                ['customers', 'customername', 'customerid'],
                ['orders', 'orderid', 'customerid']
            )
            self.assertEqual(len(c.rows('customers')), 213)


    def test_pwork(self):
        with connect(":memory:") as c:
            c.load('products.csv')
            c.pwork(price_sum, 'products', [pred1, pred2])
            self.assertEqual(len(c.rows('psum')), 8)


class TestMisc(unittest.TestCase):
    def test_isnum(self):
        self.assertEqual(isnum(3), True)
        self.assertEqual(isnum(-3.32), True)
        self.assertEqual(isnum("abc"), False)

    def test_load_excel(self):
        with connect(':memory:') as c:
            c.load('orders.xlsx', 'orders_temp')
            # You may see some surprises because
            # read_excel uses pandas way of reading excel files
            # q.rows('orders1').show()
            self.assertEqual(len(c.rows('orders_temp')), 196)

    def test_sas(self):
        with connect(':memory:') as c:
            c.load('ff5_ew_mine.sas7bdat')
            self.assertEqual(len(c.rows('ff5_ew_mine')), 253)

    def test_readxl(self):
        xs = list(readxl('comp.xlsx'))
        self.assertEqual(len(xs), 185)
        self.assertEqual(len(xs[0]), 31)


if __name__ == "__main__":
    if os.path.exists('workspace.db'):
        os.remove('workspace.db')
    process(
        Load('orders.csv'),
        # same as Load('orders.csv', name='orders')
        Load('customers.csv'),
        Map(month, 'orders', name='orders1'),
        Join(
            ['orders1', '*', 'customerid'],
            ['customers', 'customername, country', lambda r: [r.CustomerID - 1]],
            name="orders2"
        ),

        Map(cnt, 'orders2', group='yyyymm', overlap=3, arg=3, name='order_cnt'),
        Map(cnt, 'orders2', group='yyyymm', overlap=6, arg=6, name='order_cnt'),

        Map(lambda r: r, 'orders2', name='orders3'),
        Union('orders2, orders3', name='orders4'),

        Map(lambda rs: rs, 'orders', group='*', name='orders_all')

    )

    with connect('workspace.db') as c:
        assert len(allrows(c, 'order_cnt').where(lambda r: r.n == 3)) == 6
        assert len(allrows(c, 'order_cnt').where(lambda r: r.n == 6)) == 3
        assert len(allrows(c, 'orders2')) * 2 == len(allrows(c, 'orders4')) 
        assert len(allrows(c, 'orders_all'))  == len(allrows(c, 'orders'))

    unittest.main()

