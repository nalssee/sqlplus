import os
import unittest
from sqlplus import connect, Rows, Row, isnum, setdir
from itertools import chain

setdir('data')


def overlap(xs, size, step=1):
    result = []
    for i in range(0, len(xs), step):
        result.append(Rows(chain(*xs[i: i + size])))
    return result


def price_sum(dbname, where):
    with connect(dbname) as c:
        def fn():
            for rs in c.fetch('products', group='categoryid', where=where):
                r = Row()
                r.categoryid = rs[0].categoryid
                r.psum = sum(rs['price'])
                yield r
        c.insert(fn(), 'psum')


class TestConnection(unittest.TestCase):
    def test_avg_by_group(self):
        with connect('test.db') as c:
            c.load('products.csv')

            def products_avg():
                for rs in c.fetch('products', group="categoryid"):
                    r = Row()
                    r.categoryid = rs[0].categoryid
                    r.agg_price = sum(r.price for r in rs)
                    r.n = len(rs)
                    yield r
            c.insert(products_avg(), 'products_avg', pkeys='categoryid')

            self.assertEqual(c._pkeys('products_avg'), ['categoryid'])
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
            for rs in c.fetch('products', group='categoryid', overlap=3):
                x.append(rs[0].productname)
                rs['productname'] = 0
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

    def test_pwork(self):
        with connect(":memory:") as c:
            c.load('products.csv')
            c.pwork(price_sum, 'products',
                    ['categoryid < 5', 'categoryid >= 5'])
            self.assertEqual(len(c.rows('psum')), 8)

    def test_join(self):
        with connect(':memory:') as c:
            c.load('customers.csv')
            c.load('orders.csv')
            c.join(
                ['customers', 'customername', 'customerid'],
                ['orders', 'orderid', 'customerid']
            )
            self.assertEqual(len(c.rows('customers')), 213)


class TestRows(unittest.TestCase):
    def test_rows_group_and_overlap(self):
        with connect(':memory:') as c:
            c.load('products.csv')
            rss = c.rows('products').group('categoryid')
            self.assertEqual([len(rs) for rs in rss],
                             [12, 12, 13, 10, 7, 6, 5, 12])

            sizes = []
            for x in overlap(rss, 5, 2):
                sizes.append(len(x))
            self.assertEqual(sizes, [54, 41, 30, 17])


class TestMisc(unittest.TestCase):
    def test_isnum(self):
        self.assertEqual(isnum(3), True)
        self.assertEqual(isnum(-3.32), True)
        self.assertEqual(isnum("abc"), False)


if __name__ == "__main__":
    unittest.main()


