from sqlplus import *

def month(r):
    r.yyyymm = dconv(r.orderdate, "%Y-%m-%d", "%Y-%m")
    yield r


if __name__ == "__main__":
    process(
        Load('orders.csv'),
        Load('customers.csv'),
        Apply('orders', 'orders1', month),
        Join(
            ['orders1', '*', 'customerid'],
            ['customers', 'customername, country', 'customerid'],
            name="orders2"
        )
    )
    pass
