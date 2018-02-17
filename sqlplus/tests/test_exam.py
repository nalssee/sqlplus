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


# if __name__ == "__main__":
#     os.remove('workspace.db')
#     process(
#         Load('orders.csv'),
#         Load('customers.csv'),
#         Apply('orders', 'orders1', month),
#         Join(
#             ['orders1', '*', 'customerid'],
#             ['customers', 'customername, country', 'customerid'],
#             name="orders2"
#         ),

#         Apply('orders2', 'order_cnt', cnt, group='yyyymm', overlap=3, arg=3),
#         Apply('orders2', 'order_cnt', cnt, group='yyyymm', overlap=6, arg=6)
#     )

#     with connect('workspace.db') as c:
#         assert len(c.rows('order_cnt').where(lambda r: r.n == 3)) == 6
#         assert len(c.rows('order_cnt').where(lambda r: r.n == 6)) == 3




if __name__ == "__main__":
    os.remove('workspace.db')
    process(
        orders=Load('orders.csv'),
        customers=Load('customers.csv'),
        orders1=Apply('orders', month),
        orders2=Join(
                ['orders1', '*', 'customerid'],
                ['customers', 'customername, country', 'customerid'],
        ),
        orders_cnt=[Apply('orders2', cnt, group='yyyymm', overlap=3, arg=3),
                    Apply('orders2', cnt, group='yyyymm', overlap=6, arg=6)]
    )

    with connect('workspace.db') as c:
        assert len(c.rows('order_cnt').where(lambda r: r.n == 3)) == 6
        assert len(c.rows('order_cnt').where(lambda r: r.n == 6)) == 3




