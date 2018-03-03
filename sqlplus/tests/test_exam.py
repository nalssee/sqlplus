import os
from sqlplus import *


def month(r):
    r.yyyymm = dconv(r.orderdate, "%Y-%m-%d", "%Y-%m")
    yield r


def addmonth(date, n):
    return dmath(date, f"{n} months", "%Y-%m")


def cnt(rs, n):
    if addmonth(rs[0].yyyymm, n - 1) == rs[-1].yyyymm:
        r = Row()
        r.yyyymm = rs[-1].yyyymm
        r.cnt = len(rs)
        r.n = n
        yield r


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

        Map(lambda r: [r], 'orders2', name='orders3'),
        Union('orders2, orders3', name='orders4')

    )

    with connect('workspace.db') as c:
        assert len(c.rows('order_cnt').where(lambda r: r.n == 3)) == 6
        assert len(c.rows('order_cnt').where(lambda r: r.n == 6)) == 3

        assert len(c.rows('orders2')) * 2 == len(c.rows('orders4')) 





