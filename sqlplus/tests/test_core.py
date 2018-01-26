import os
import sys
import unittest
from itertools import chain

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from sqlplus import connect, Rows, Row, isnum, setdir, dconv, dmath


setdir('data')


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
    def test_isnum(self):
        with connect(':memory:') as q:
            q.load('customers.csv')
            rs1 = q.rows('customers', where='isnum(PostalCode)')
            rs2 = q.rows('customers').isnum('PostalCode')
            self.assertEqual(len(rs1), len(rs2))

            rs1 = q.rows('customers', where='isnum(PostalCode, CustomerID)')
            rs2 = q.rows('customers', where='isnum(PostalCode)')
            rs3 = q.rows('customers', where='isnum(PostalCode, City)')
            self.assertEqual(len(rs1), 66)
            self.assertEqual(len(rs2), 66)
            self.assertEqual(len(rs3), 0)

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
            c.insert(products_avg(), 'products_avg', pkeys='CategoryID')

            self.assertEqual(c._pkeys('products_avg'), ['CategoryID'])
            self.assertEqual(c.rows('products_avg')['agg_price'],
                             [455.75, 276.75, 327.08, 287.3,
                              141.75, 324.04, 161.85, 248.19])
            self.assertEqual(c.rows('products_avg')['n'],
                             [12, 12, 13, 10, 7, 6, 5, 12])

        os.remove('data/test.db')

    def test_group_and_overlap(self):
        with connect(':memory:') as c:
            c.load('orders.csv')
            c.create("""select *, dconv(orderdate, '%Y-%m-%d', '%Y-%m') as ym
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
                rs.assign('ProductName', 0)
            # !!!!! overlap shares rows in between iterations
            self.assertEqual(x, ['Chais', 0, 0, 0, 0, 0, 0, 0])

    def test_to_csv(self):
        with connect(':memory:') as c:
            c.load('orders.csv')
            c.to_csv('orders', 'orders1.csv')
            c.drop('orders')
            c.load('orders1.csv')
            self.assertEqual(len(c.rows('orders1')), 196)
        os.remove('data/orders1.csv')

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
            c.pwork(price_sum, 'products',
                    ['CategoryID < 5', 'CategoryID >= 5'])
            self.assertEqual(len(c.rows('psum')), 8)

    def test_register(self):
        def product(xs):
            result = 1
            for x in xs:
                result *= x
            return result

        with connect(':memory:') as c:
            def foo(x, y):
                return x + y

            def bar(*args):
                return sum(args)

            #
            def foo1(a, b):
                sum = 0
                for a1, b1 in zip(a, b):
                    sum += a1 * b1
                return sum

            def bar1(*args):
                sum = 0
                for xs in zip(*args):
                    sum += product(xs)
                return sum

            c.register(foo)
            c.register(bar)
            # Look up the def of 'foo1' and you'll see r.a and r.b
            # Actual table doesn't have to have column a and b
            c.register_agg(foo1)
            c.register_agg(bar1)

            c.sql("create table test(i, j, x)")
            c.sql("insert into test values (1, 3,'a')")
            c.sql("insert into test values (21, 2, 'b')")
            c.sql("insert into test values (5,3, 'a')")
            c.sql("insert into test values (20,4, 'a')")
            c.sql("insert into test values (20,'x', 'c')")
            c.sql("insert into test values (20,-1.2, 'd')")

            c.create("select foo(i, j) as val1, bar(i, j) as val2 from test",
                     'test1')
            self.assertEqual(c.rows('test1')['val1'], [4, 23, 8, 24, '', 18.8])
            self.assertEqual(c.rows('test1')['val2'], [4, 23, 8, 24, '', 18.8])

            c.create("""
            select foo1(i, j) as val1, bar1(i, j) as val2 from test group by x
            """, 'test2')
            self.assertEqual(c.rows('test2')['val1'], [98, 42, '', -24.0])
            self.assertEqual(c.rows('test2')['val2'], [98, 42, '', -24.0])


    def test_join2(self):
        def avg_id(rs):
            r = Row(date=dconv(rs[0].orderdate, '%Y-%m-%d', '%Y%m'))
            r.orderid = round(rs.avg('orderid'))
            r.customerid = round(rs.avg('orderid'))
            r.employeeid = round(rs.avg('employeeid'))
            r.shipperid = rs[0].shipperid
            return r

        with connect(':memory:') as q:
            q.load('customers.csv')
            q.load('orders.csv')

            q.join(
                ['customers', 'customername', 'customerid'],
                # if the matching columns (the third item in the following list
                # is missing, then it is assumed to be the same as
                # the matching column of the first table
                ['orders', 'orderid'],
                name='customers1'
            )
            rs = q.rows('customers1')
            self.assertEqual(len(rs), 213)
            self.assertEqual(len(rs.isnum('orderid')), 196)
            q.drop('customers1')

            def to_month(r):
                r.date = dconv(r.orderdate, '%Y-%m-%d', '%Y%m')
                return r
            tseq = (to_month(r) for r in q.fetch('orders'))
            q.insert(tseq, 'orders1')
            # There's no benefits in using multiple cores
            # You should know what you are doing.

            tseq = (avg_id(r) for r in q.fetch('orders1', group='date'))
            q.insert(tseq, 'orders2')

            # testing reel
            ls = []
            for rs in q.fetch('orders2', group='date', overlap=(5, 2)):
                ls.append(len(rs))
            self.assertEqual(ls, [5, 5, 4, 2])

            self.assertEqual(len(q.rows('orders1')), 196)

            tseq = (rs[0] for rs in q.fetch('orders1',
                                            group='date, customerid'))
            q.insert(tseq, 'orders3', pkeys='date, customerid')
            self.assertEqual(len(q.rows('orders3')), 161)

            def addm(date, n):
                return dmath(date, {'months': n}, '%Y%m')

            q.register(addm)
            q.create('select *, addm(date, 1) as d1 from orders1', 'orders1_1')
            q.create('select *, addm(date, 2) as d2 from orders1', 'orders1_2')
            q.create('select *, addm(date, 3) as d3 from orders1', 'orders1_3')
            q.join(
                ['orders1', 'date, customerid, orderid', 'date, customerid'],
                ['orders1_1', 'orderid as orderid1', 'd1, customerid'],
                ['orders1_2', 'orderid as orderid2', 'd2, customerid'],
                ['orders1_3', 'orderid as orderid3', 'd3, customerid'],
                name='orders3'
            )
            q.drop('orders1_1, orders1_2, orders1_3')

            q.create("""
            select a.date, a.customerid, a.orderid,
            b.orderid as orderid1,
            c.orderid as orderid2,
            d.orderid as orderid3

            from orders1 as a

            left join orders1 as b
            on a.date = addm(b.date, 1) and a.customerid = b.customerid

            left join orders1 as c
            on a.date = addm(c.date, 2) and a.customerid = c.customerid

            left join orders1 as d
            on a.date = addm(d.date, 3) and a.customerid = d.customerid
            """, name='orders4')

            rs3 = q.rows('orders3')
            rs4 = q.rows('orders4')

            for r3, r4 in zip(rs3, rs4):
                self.assertEqual(r3.values, r4.values)


class TestMisc(unittest.TestCase):
    def test_isnum(self):
        self.assertEqual(isnum(3), True)
        self.assertEqual(isnum(-3.32), True)
        self.assertEqual(isnum("abc"), False)

    def test_dmath_and_dconv(self):
        with connect(':memory:') as c:
            c.load('orders.csv')
            c.create("""
            select *, dmath(orderdate, "2 month", "%Y-%m-%d") as date
            from orders""", 'orders1')

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


if __name__ == "__main__":
    unittest.main()


