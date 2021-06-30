import os
import shutil
import pytest
import sqlplus as sp 

# much more work to do

@pytest.fixture(autouse=True)
def run_around_tests():
    files = ['pytest.db', 
             'pytest.gv', 
             'pytest.gv.pdf']
    def remfiles():
        for fname in files:
            if os.path.isfile(fname):
                os.remove(fname)
        sp.sqlplus._JOBS = {}
        shutil.rmtree(sp.sqlplus._TEMP, ignore_errors=True)
    remfiles()
    yield
    remfiles()


def test_loading_ordinary_csv():
    sp.register(orders=sp.load('tests/Orders.csv'))
    sp.run()
    orders1 = sp.get('orders')
    orders2 = sp.util.readxl('tests/Orders.csv')
    header = next(orders2)
    assert list(orders1[0].keys()) == header 

    for a, b in zip(orders1, orders2):
        assert list(a.values()) == b


def test_apply_order_year():
    def year(r):
        r['order_year'] = r['order_date'][:4]
        return r

    sp.register(
        orders=sp.load('tests/Orders.csv'),
        orders1=sp.apply(year, 'orders')
    )

    sp.run()
    for r in sp.get('orders1'):
        assert r['order_year'] == int(r['order_date'][:4])


def test_apply_group1():
    def size(rs):
        r0 = rs[0]
        r0['n'] = len(rs)
        return r0

    sp.register(
        order_items=sp.load('tests/OrderItems.csv'),
        order_items1=sp.apply(size, 'order_items', by='prod_id'),
        order_items2=sp.apply(size, 'order_items', by='prod_id, order_item'),
    )
    sp.run()
    assert len(sp.get('order_items1')) == 7
    assert len(sp.get('order_items2')) == 16 


def test_join():
    sp.register(
        products = sp.load('tests/Products.csv'),
        vendors = sp.load('tests/Vendors.csv'),
        products1 = sp.join(
            ['products', '*', 'vend_id'],
            ['vendors', 'vend_name, vend_country', 'vend_id'],
        )
    )
    sp.run()
    products1 = sp.get('products1')
    assert products1[0]['vend_country'] == 'USA'
    assert products1[-1]['vend_country'] == 'England'


def test_parallel1():
    def revenue(r):
        r['rev'] = r['quantity'] * r['item_price']
        return r

    sp.register(
        items = sp.load('tests/OrderItems.csv'),
        items1 = sp.apply(revenue, 'items'),
        items2 = sp.apply(revenue, 'items', parallel=True),

    )
    sp.run()
    assert sp.get('items1') == sp.get('items2')


def test_parallel2():
    def size(rs):
        r0 = rs[0]
        r0['n'] = len(rs)
        return r0

    sp.register(
        items = sp.load('tests/OrderItems.csv'),
        items1 = sp.apply(size, 'items', by='prod_id'),
        items2 = sp.apply(size, 'items', by='prod_id', parallel=True),

    )
    sp.run()
    assert sp.get('items1') == sp.get('items2')

