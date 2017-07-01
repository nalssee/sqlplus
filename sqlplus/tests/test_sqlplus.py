import os
import sys
import unittest
from itertools import islice
import time
import statistics as st

from scipy.stats import ttest_1samp

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)
from sqlplus.core import *
from sqlplus.util import mpairs, isnum,  \
    prepend_header, pmap, grouper, same, ymd, read_date, listify

from sqlplus.load import fnguide, sas

def fib(n):
    if n < 2:
        return n
    else:
        return fib(n - 1) + fib(n - 2)

def fn(x, y):
    return fib(x + y)

def mean0(seq):
    return round(st.mean(seq), 3)

def mean1(seq):
    "sequence of numbers with t val"
    tstat = ttest_1samp(seq, 0)
    return "%s [%s]" % (star(st.mean(seq), tstat[1]), round(tstat[0], 3))

def star(val, pval):
    "put stars according to p-value"
    if pval < 0.001:
        return str(round(val, 3)) + '***'
    elif pval < 0.01:
        return str(round(val, 3)) + '**'
    elif pval < 0.05:
        return str(round(val, 3)) + '*'
    else:
        return str(round(val, 3))


setenv('data')

class Testdbopen(unittest.TestCase):

    def test_gby(self):
       with dbopen(':memory:') as c:
            # if the string ends with '.csv'
            c.show('iris.csv', 5)

            def first_char(r):
                # doesn't matter if you return or not
                r.sp1 = r.species[:1]
                return r

            c.save('iris.csv', fn=first_char, name='first_char')

            def top20_sl():
                for rs in c.reel(
                    "select * from first_char order by sp1, sepal_length desc",
                    group='sp1'):
                    yield from rs[:20]

            c.save(top20_sl)

            print("\nYou should see the same two tables")
            print("==========")
            c.show("select col, sepal_length from top20_sl", n=3)
            print("----------")
            c.show(top20_sl, n=3, cols='col, sepal_length')
            print("==========")

            r0, r1 = list(
                c.reel("""select avg(sepal_length) as slavg
                from top20_sl group by sp1
                """))
            self.assertEqual(round(r0.slavg, 3), 5.335)
            self.assertEqual(round(r1.slavg, 3), 7.235)

            # gby with empty list group
            # All of the rows in a table is grouped.
            self.assertEqual(len(list(c.reel('first_char'))), 150)
            # list_tables, in alphabetical order
            self.assertEqual(c.tables, ['first_char', 'top20_sl'])

            # get the whole rows
            for rs in c.reel('top20_sl', group=lambda r: 1):
                self.assertEqual(len(rs), 40)

    def test_run_over_run(self):
        with dbopen(':memory:') as conn:
            conn.save("iris.csv", name="iris1")
            conn.save("iris.csv", name="iris2")
            a = conn.reel("select * from iris1 where species='setosa'")
            b = conn.reel("select * from iris2 where species='versicolor'")
            self.assertEqual(next(a).species, 'setosa')
            self.assertEqual(next(b).species, 'versicolor')
            # now you iterate over 'a' again and you may expect 'setosa'
            # to show up
            # but you'll see 'versicolor'
            # it doesn't matter you iterate over a or b
            # you simply iterate over the most recent query.
            self.assertEqual(next(a).species, 'versicolor')
            self.assertEqual(next(b).species, 'versicolor')

    def test_del(self):
        """tests column deletion
        """
        with dbopen(':memory:') as conn:
            conn.save('co2.csv')

            def co2_less(*col):
                """remove columns"""
                for r in conn.reel('co2'):
                    for c in col:
                        del r[c]
                    yield r
            print('\nco2 table')
            print('=============================================')
            conn.show("co2", n=2)
            print('=============================================')
            print("\nco2 table without plant and conc")
            print('=============================================')

            conn.show(co2_less('plant', 'conc'), n=2)
            print('=============================================')
            conn.save(co2_less('plant', 'conc'), 'co2_less')

    def test_saving_csv(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            iris = c.reel('iris', group='species')

            def first2group():
                for rs in islice(iris, 2):
                    yield from rs

            c.show(first2group, file='sample.csv')
            self.assertTrue(os.path.isfile(os.path.join('data', 'sample.csv')))

            # each group contains 50 rows, hence 100
            c.save('sample.csv')
            self.assertEqual(len(list(c.reel('sample'))), 100)
            os.remove(os.path.join('data', 'sample.csv'))

    def test_column_case(self):
        with dbopen(':memory:') as conn:
            conn.run("create table Foo (a int, B real)")
            conn.run("insert into foo values (10, 20.2)")
            # table name is case-insensitive
            rows = list(conn.reel('foO'))
            # but columns names are at least in my system, OS X El Capitan
            # I don't know well about it
            self.assertEqual(rows[0].B, 20.2)

    def test_order_of_columns(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            row = next(c.reel('iris'))
            self.assertEqual(row.columns,
                             ['col', 'sepal_length', 'sepal_width',
                              'petal_length', 'petal_width', 'species'])

    def test_unsafe_save(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            def unsafe():
                for rs in c.reel('iris', group='species'):
                    rs[0].a = 'a'
                    yield rs[0]
                    for r in rs[1:]:
                        r.b = 'b'
                        yield r
            # when rows are not alike, you can't save it
            with self.assertRaises(Exception):
                conn.save(unsafe)

    def test_todf(self):
        with dbopen(':memory:') as conn:
            conn.save('iris.csv')
            for rs in conn.reel('iris', group='species'):
                self.assertEqual(rs.df().shape, (50, 6))


class TestRow(unittest.TestCase):
    def test_row(self):
        r1 = Row()
        self.assertEqual(r1.columns, [])
        self.assertEqual(r1.values, [])

        r1.x = 10
        r1.y = 'abc'
        r1.z = 39.2

        self.assertEqual(r1.copy().columns, ['x', 'y', 'z'])
        self.assertEqual(r1.copy().values, [r1.x, r1.y, r1.z])

        self.assertEqual(r1.columns, ['x', 'y', 'z'])
        self.assertEqual(r1.values, [10, 'abc', 39.2])

        with self.assertRaises(Exception):
            r1.a

        with self.assertRaises(Exception):
            del r1.a

        del r1.y

        self.assertEqual(r1.columns, ['x', 'z'])
        self.assertEqual(r1.values, [10, 39.2])

        r1.x *= 10
        r1.z = r1.x - r1.z
        self.assertEqual(r1.values, [r1.x, r1.z])

    def test_row2(self):
        r1 = Row()
        self.assertEqual(r1.columns, [])
        self.assertEqual(r1.values, [])

        r1['x'] = 10
        r1['y'] = 'abc'
        r1['z'] = 39.2

        self.assertEqual(r1.columns, ['x', 'y', 'z'])
        self.assertEqual(r1.values, [10, 'abc', 39.2])

        with self.assertRaises(Exception):
            r1['a']

        with self.assertRaises(Exception):
            del r1['a']

        del r1['y']

        self.assertEqual(r1.columns, ['x', 'z'])
        self.assertEqual(r1.values, [10, 39.2])

        r1['x'] *= 10
        r1['z'] = r1['x'] - r1['z']
        self.assertEqual(r1.values, [r1['x'], r1['z']])

    def test_row3(self):
        r1 = Row(x=10, y=20, z=30, w=40)
        # order must be kept
        self.assertEqual(r1.columns, ['x', 'y', 'z', 'w'])


class TestMisc(unittest.TestCase):
    def test_sample(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            c.desc('iris')

    def test_prepend_header(self):
        # since prepend_header is a util you need to pass the full path
        iris2 = os.path.join('data', 'iris2.csv')
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            c.show('iris', file='iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=20)
            c.save('iris2.csv')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 20)

            c.show(c.reel('iris'), file='iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=1)
            c.save('iris2.csv')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 1)

            c.show(c.reel('iris'), file='iris2.csv')
            prepend_header(iris2, 'cnt, sl, sw, pl, pw, sp', drop=0)
            c.save('iris2.csv')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 'col')
            self.assertEqual(first.sl, 'sepal_length')

            c.show(c.reel('iris'), file='iris2.csv')
            # simply drop the first 5 lines, and do nothing else
            prepend_header(iris2, header=None, drop=5)
            # don't drop any and just write the header
            prepend_header(iris2, header='cnt, sl, sw, pl, pw, sp', drop=0)
            c.save('iris2.csv')
            first = next(c.reel('iris2'))
            self.assertEqual(first.cnt, 5)

            os.remove(iris2)

    def test_dup_columns(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')

            with self.assertRaises(Exception):
                # there can't be duplicates
                next(c.reel("""
                select species, sepal_length, sepal_length
                from iris
                """))

            r1 = next(c.reel("""
                      select species, sepal_length,
                      sepal_length as sepal_length10
                      from iris
                      """))
            r1.sepal_length10 = r1.sepal_length * 10
            self.assertEqual(r1.values, ['setosa', 5.1, 51.0])

    def test_utilfns(self):
        self.assertTrue(isnum(3))
        self.assertTrue(isnum(-3.32))
        self.assertFalse(isnum('32.3'))

        self.assertEqual(ymd(199912, 2), 200002)
        self.assertEqual(ymd(199912, -2), 199910)

        self.assertEqual(ymd(19991231, 2), 20000102)
        self.assertEqual(ymd(19991231, -2), 19991229)


class TestMisc2(unittest.TestCase):
    def test_save_with_implicit_name(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            c.save('select * from iris where species="setosa"')
            self.assertEqual(len(c.rows('iris')), 50)

    def test_pmap(self):
        result = []
        for x in pmap(fn, list(range(10)), list(range(10, 1, -1))):
            result.append(x)
        self.assertEqual(result, [55] * 9)

    def test_fnguide(self):
        cnt = 0
        for r in fnguide('acc2.csv', 'share, debt, lt, cf'):
            if cnt >= 10:
                break
            print(r)
            cnt += 1


    def test_sas(self):
        cnt = 0
        rs = []
        for r in sas('ff5_ew_mine.sas7bdat'):
            if cnt >= 3:
                break
            rs.append(r)
            cnt += 1
        cols = ['date', 'smb', 'yymm', 'smbinv',
                'smbop', 'hml', 'rmw', 'cma', 'mktret',
                'riskfree', 'smbn', 'rf']
        self.assertEqual(rs[0].columns, cols)

    def test_single(self):
        rs = []
        for i in range(10):
            r = Row(date=ymd(2001, i), even=i%2)
            rs.append(r)
        rs.append(Row(date=2003, even=False))
        rs.append(Row(date=2008, even=True))
        # intentional mistake
        rs.append(Row(date=2003, even=True))

        with dbopen(':memory:') as c:
            with self.assertRaises(Exception):
                # DATE(and ID if exsits) is a primary key
                # so you can't insert rs
                c.save(Rows(rs), 'foo')
            rs1 = []
            for r in rs:
                r.date1 = r.date
                del r.date
                rs1.append(r)

            c.save(Rows(rs1), 'foo')

            a = list(c.single('foo', dup=True))
            b = list(c.single('foo'))
            c1 = list(c.single('foo', dup=True, cols='date1'))
            d = list(c.single('foo', cols='even'))
            self.assertEqual(len(a),  4)
            self.assertEqual(len(b),  11)
            self.assertEqual(len(c1),  5)
            self.assertEqual(len(d),  2)

    def test_read_date(self):
        self.assertEqual(read_date('31Mar2013', '%d%b%Y'), '20130331')
        self.assertEqual(read_date('31-Mar-2013', '%d-%b-%Y', "%Y%m"), '201303')
        self.assertEqual(read_date('Mar2013', "%b%Y", "%Y%m"), '201303')



class TestRows(unittest.TestCase):
    def test_ljoin(self):
        def seq(*args):
            return Rows(Row(date=arg) for arg in args)
        rs0 = Rows(Row(date=ymd(2000, i)) for i in range(10))
        rs1 = seq(2001, 2003, 2008, 2009)
        rs2 = seq(2003, 2005, 2007, 2012)
        ys = [[x.date if x else None for x in xs] for xs in rs0.ljoin(rs1, rs2)]
        self.assertEqual(ys[7], [2007, None, 2007])
        self.assertEqual(ys[8], [2008, 2008, None])
        for y in ys:
            self.assertEqual(len(set(y1 for y1 in y if y1)), 1)

        xs = []
        for x in rs1.ljoin(rs0, rs2):
            xs.append([x1['date'] for x1 in x if x1])
        self.assertEqual(xs, [[2001, 2001], [2003, 2003, 2003], [2008, 2008], [2009, 2009]])
        xs = []
        for x in rs2.ljoin(rs0, rs1):
            xs.append([x1['date'] for x1 in x if x1])
        self.assertEqual(xs, [[2003, 2003, 2003], [2005, 2005], [2007, 2007], [2012]])

    def test_rows1(self):

        with dbopen(':memory:') as c:
            c.save('iris.csv')

            iris = c.rows('iris')
            # rows must be iterable
            self.assertEqual(sum(1 for _ in iris), 150)

            self.assertTrue(isinstance(iris[0], Row))
            self.assertTrue(hasattr(iris[2:3], 'order'))
            # hasattr doesn't work correctly for Row

            self.assertFalse('order' in dir(iris[2]))
            del iris[3:]
            self.assertTrue(hasattr(iris, 'order'))
            self.assertEqual(iris['sepal_length, sepal_width, species'][2][2],
                             'setosa')
            iris['one, two'] = [[1, 2] for _ in range(3)]
            self.assertEqual(iris['one, two'], [[1, 2] for _ in range(3)])
            del iris['one, col']
            self.assertTrue(hasattr(iris, 'order'))
            self.assertEqual(len(iris[0].columns), 6)

            # append heterogeneuos row
            iris.append(Row())
            with self.assertRaises(Exception):
                iris.df()
            iris[:3].df()

            iris.append(Row())
            with self.assertRaises(Exception):
                c.save(iris, 'iris_sample')
            c.save(iris[:3], 'iris_sample')


    def test_rows2(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            iris = c.rows('iris')
            # order is destructive
            iris.order('sepal_length, sepal_width', reverse=True)
            self.assertEqual(iris[0].col, 132)
            self.assertEqual(iris[1].col, 118)
            self.assertEqual(iris[2].col, 136)

            col1 = iris.where(lambda r: r['species'] == 'versicolor')[0].col

            self.assertEqual(col1, 51)
            # where is non-destructive
            self.assertEqual(iris[0].col, 132)

            iris = c.rows('iris')
            iris.order('sepal_length, sepal_width', reverse=True)

            self.assertEqual(len(next(iris.group('species'))), 12)
            # just because..
            sum = 0
            for rs in iris.group('species'):
                sum += len(rs)
            self.assertEqual(sum, 150)

            iris = c.rows('iris')

            self.assertEqual(len(iris.num('species')), 0)
            self.assertEqual(len(iris.text('species')), 150)

            self.assertEqual(len(iris.num('sepal_length, sepal_width')), 150)
            self.assertEqual(len(iris.where(lambda r: r.species in
                                            ['versicolor', 'virginica'])),
                             100)
            self.assertEqual(len(iris.where(lambda r: r.sepal_length == 5.0)), 10)

            rs = []
            for x in range(10):
                rs.append(Row(x=x))
            c.save(rs, 'temp')
            rs = c.rows('temp')
            self.assertEqual(rs.truncate('x', 0.2)['x'],
                             [2, 3, 4, 5, 6, 7])
            self.assertEqual([int(x * 10) for x in rs.order('x', True).winsorize('x', 0.2)['x']],
                             [72, 72, 70, 60, 50, 40, 30, 20, 18, 18])


    def test_rows3(self):
        rs = Rows([Row(), Row(), Row()])
        rs['a'] = 10
        self.assertEqual(rs['a'], [10, 10, 10])
        rs['a, b'] = [3, 4]
        self.assertEqual(rs['a, b'], [[3, 4], [3, 4], [3, 4]])
        with self.assertRaises(Exception):
            rs['a, b'] = [3, 4, 5]
        rs[1:]['a, b'] = [[1, 2], [3, 5]]
        self.assertEqual(rs['a, b'], [[3, 4], [1, 2], [3, 5]])
        with self.assertRaises(Exception):
            rs['a, b'] = [[1, 2], [3, 40], [10, 100, 100]]

        self.assertEqual(int(rs.wavg('a') * 1000), 2333)
        self.assertEqual(int(rs.wavg('a', 'b') * 1000), 2636)

    def test_describe(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            iris = c.rows('iris')
            self.assertTrue('petal_width' in iris[0].columns)
            for g in iris.group('species'):
                df = g.df('sepal_length, sepal_width')
                summary = df.describe()
                self.assertFalse('petal_width' in dir(summary))
                # you can plot it
                # df.plot.scatter(x='sepal_length', y='sepal_width')
                # plt.show()


class TestUserDefinedFunctions(unittest.TestCase):
    def test_simple(self):
        # isnum, istext, yyyymm
        with dbopen(':memory:') as c:
            # fama french 5 industry portfolios
            c.save('indport.csv')
            c.run(
                """
                create table if not exists indport1 as
                select *, substr(date, 1, 4) as yyyy,
                substr(date, 1, 6) as yyyymm,
                case
                when cnsmr >= 0 then 1
                else 'neg'
                end as sign_cnsmr
                from indport
                """)

            na = len(c.rows("select * from indport1 where isnum(sign_cnsmr)"))
            nb = len(c.rows("select * from indport1 where not isnum(sign_cnsmr)"))
            nc = len(c.rows("select * from indport1"))
            self.assertEqual(na + nb, nc)

            r = next(c.reel(
                """
                select *, ymd(substr(date, 1, 6), 12) as yyyymm1
                from indport
                where date >= 20160801
                """))
            self.assertEqual(r.yyyymm1, 201708)


class TestMpairs(unittest.TestCase):
    def test_mpairs(self):
        # lists, iterators are OK
        xs = iter([2, 4, 7, 9, 10, 11, 21])
        ys = [1, 3, 4, 9, 10, 21, 100]
        result = []
        for a, b in mpairs(xs, ys, lambda x: x):
            result.append(a)
        self.assertEqual(result, [4, 9, 10, 21])

    def test_mpairs_with_double_dbs(self):
        with dbopen(':memory:') as c1, dbopen(':memory:') as c2:
            c1.save('iris.csv')
            c2.save('co2.csv')

            seq1 = c1.reel(
                """
                select *, round(petal_length) as key from iris
                order by key
                """, group='key')
            seq2 = c2.reel(
                """
                select *, round(uptake / 10) as key from co2
                order by key
                """, group='key')

            lengths = []

            for a, b in mpairs(seq1, seq2, lambda rs: rs[0].key):
                lengths.append((len(a), len(b)))

            self.assertEqual(lengths, [(24, 18), (26, 17),
                                       (3, 24), (26, 24), (43, 1)])


class TestOLS(unittest.TestCase):
    def test_ols(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            for rs in c.reel('iris', group='species'):
                result = rs.ols('sepal_width ~ petal_width + petal_length')
                # maybe you should test more here
                self.assertEqual(result.nobs, 50)
                self.assertEqual(len(result.params), 3)
            # Rows(c.reel('iris')).plot('sepal_length, petal_width, petal_length')
            # c.plot('iris', 'sepal_length, sepal_width')

    def test_reg(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            for rs in c.reel('iris', group='species'):
                rs.reg('sepal_width ~ petal_width + petal_length').show()


class TestDFMisc(unittest.TestCase):
    def test_describe(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            print("IRIS")
            c.desc("""select sepal_length, sepal_width, petal_length,
                   petal_width from iris""")


# Test methods for portfolio works
class TestPort(unittest.TestCase):
    def setUp(self):
        rs1 = []
        for year in range(2001, 2011):
            rs1.append(Row(date=year))
        self.rs1 = Rows(rs1)

        rs2 = []
        start_month = 200101
        for i in range(36):
            rs2.append(Row(date=ymd(start_month, i)))
        self.rs2 = Rows(rs2)

        rs3 = []
        start_date = 20010101
        for i in range(30):
            rs3.append(Row(date=ymd(start_date, i)))
        self.rs3 = Rows(rs3)

        with dbopen(':memory:') as c:
            c.save('indport.csv')
            # to pseudo monthly data
            rs = []
            for rs1 in c.reel('indport order by date', group=lambda r: str(r.date)[0:4]):
                for r in rs1:
                    date = str(r.date)
                    r.date = int(date[:4])
                    r.id = 'A' + date[4:]
                    rs.append(r)
            self.indport = Rows(rs)

    def test_indi_sort0(self):
        # shoudl see some alerts but works fine. Of course you can turn off it (it's default!!)
        self.indport.pn(cnsmr=2, manuf=3, alert=True).pavg('other').pat('other')

        self.indport.between(2007,  2010).pn(cnsmr=2, manuf=3).pavg('other').pat('other')
        with self.assertRaises(Exception):
            # 2008, 2009 (betwwen 2007 <= x < 2009)
            # but 2009 has missing portfolio as you've seen above
            self.indport.between(2008, 2010).pn(cnsmr=2, manuf=3).pavg('other').pat('other')

        x = Rows([Row(date=2001, id='y', x=3), Row(date=2001, id='z', x=10), Row(date=2002, id='x', x=1),
                  Row(date=2002, id='y', x=2),Row(date=2002, id='z', x=4)])

        # It looks OK but not...see bps in util.py, check break points
        y = x.pn(x = 2, alert=True)
        with self.assertRaises(Exception):
            y.pavg('x').pat('x').csv()


    def test_indi_sort1(self):

        avgport = self.indport.where(lambda r: r.date < 2009).pn(cnsmr=2, manuf=3).pavg('other')
        ap = avgport.order(['date', 'pn_cnsmr', 'pn_manuf'])\
             .where(lambda r: r.pn_cnsmr > 0 and r.pn_manuf > 0)
        self.assertEqual(ap[0].n, 75)
        self.assertEqual(ap[1].n, 44)
        self.assertEqual(ap[2].n, 4)
        self.assertEqual(ap[3].n, 7)
        self.assertEqual(ap[4].n, 38)
        self.assertEqual(ap[5].n, 80)

        self.assertEqual(round(ap[0].other, 2), -0.64)

        indport = self.indport.where(lambda r: r.date < 2009)
        indport.pn(indport.breaks('cnsmr', 10))
        indport.where('date', 2008).show()

        other1 = []
        for year in range(2001, 2009):
            other1.append(mean0(indport.where('pn_cnsmr', 1, 'date', year)['other']))

        self.assertEqual(other1, [-1.277, -1.424, -0.859, -1.002, -0.963, -1.036, -1.799, -4.225])

        indport = self.indport.where(lambda r: r.date < 2009).pn('cnsmr', 10)
        other10 = []
        for year in range(2001, 2009):
            other10.append(mean0(indport.where('pn_cnsmr', 10, 'date', year)['other']))

        self.assertEqual(other10, [1.399, 1.483, 1.234, 0.954, 1.04, 1.165, 1.313, 3.975])

        indport.pn('cnsmr', 10)
        pat = indport.pavg('other', pncols='pn_cnsmr').pat('other', pncols='pn_cnsmr')
        self.assertEqual(round(st.mean(other10) - st.mean(other1), 2), float(pat.lines[1][11][:4]))


        indport = self.indport.between(2004, 2009).pn('cnsmr', 2, 'manuf', 3)
        other21 = []
        other23 = []
        for year in range(2004, 2009):
            pavg1 = indport.where('pn_cnsmr', 2, 'pn_manuf', 1, 'date', year)['other']
            pavg2 = indport.where('pn_cnsmr', 2, 'pn_manuf', 3, 'date', year)['other']
            other21.append(st.mean(pavg1))
            other23.append(st.mean(pavg2))

        pat = indport.pavg('other').pat('other').lines

        self.assertEqual(round(st.mean(other21), 3), float(pat[2][1].replace('*', '').split()[0]))
        self.assertEqual(round(st.mean(other23), 3), float(pat[2][3].replace('*', '').split()[0]))
        self.assertEqual(round(st.mean(other23) - st.mean(other21), 3), float(pat[2][4][:5]))

        # # # test for avgs columns!!
        pat1 = indport.pn('cnsmr', 2, 'manuf', 3)\
               .pavg('other', pncols='pn_cnsmr, pn_manuf')\
               .pat('other', pncols='pn_cnsmr, pn_manuf').lines

        pat2 = indport.pn('cnsmr', 2).pavg('other', pncols='pn_cnsmr')\
                      .pat('other', pncols='pn_cnsmr').lines

        pat3 = indport.pn('manuf', 3).pavg('other', pncols='pn_manuf')\
                      .pat('other', pncols='pn_manuf').lines

        self.assertEqual(pat2[1][1:], [x[-1] for x in pat1][1:])
        self.assertEqual(pat1[-1][1:], pat3[1][1:])


    def test_indi_sort2(self):
        "weighted average"
        avgport = self.indport.where(lambda r: r.date <= 2015).pn('cnsmr', 10)
        hlth = avgport.where('date', 2001, 'pn_cnsmr', 3)['hlth']
        other = avgport.where('date', 2001, 'pn_cnsmr', 3)['other']

        total = sum(hlth)
        result = []
        for x, y in zip(other, hlth):
            result.append(x * y / total)
        self.assertEqual(sum(result),
                         avgport.pavg('other', 'hlth')\
                         .where('date', 2001, 'pn_cnsmr', 3)['other'][0])

    def test_indi_sort3(self):
        self.assertEqual(self.indport.pn('cnsmr', 2).pavg('other').pat('other').lines,
                         self.indport.pn('cnsmr', [0.5]).pavg('other').pat('other').lines)

    def test_dpn(self):
        avgport = self.indport.pn(cnsmr=4, manuf=3, hlth=2, dependent=True).pavg('other')

        avgport1 = avgport.where(lambda r: r.pn_cnsmr != 0 and r.pn_manuf != 0 and r.pn_hlth != 0)
        for r in avgport1.where(lambda r: r.date < 2016):
            # must be about the same size
            self.assertTrue(r.n >= 10 or r.n < 14)

        seq1 = avgport.where(pn_cnsmr=3, pn_manuf=1, pn_hlth=2)['other']
        seq2 = avgport.where(pn_cnsmr=3, pn_manuf=3, pn_hlth=2)['other']

        pat = avgport.where(pn_cnsmr=3).pat('other', pncols='pn_manuf, pn_hlth').lines

        self.assertEqual(round(st.mean(seq1), 3), float(pat[1][2].split()[0].replace('*', '')))
        self.assertEqual(round(st.mean(seq2), 3), float(pat[3][2].split()[0].replace('*', '')))
        self.assertEqual(round(st.mean(seq2) - st.mean(seq1), 3), float(pat[4][2][:5]))


    def test_pnroll(self):
        a = self.indport.between(2003).pn(cnsmr=5, manuf=4, jump=5)
        for rs in a.roll(5, 5):
            for rs1 in rs.order('id, date').group('id'):
                self.assertTrue(same(rs1['pn_cnsmr, pn_manuf']))

        # not a good style, but works
        # a = self.indport.between(2003).pn(cnsmr=5, manuf=4, hi_tec=3, jump=5, warnings=True)
        # better write as follows
        a = self.indport.between(2003).pn('cnsmr', 5, 'manuf', 4, 'hi_tec', 3, jump=5)

        for rs in a.roll(5, 5):
            for rs1 in rs.order('id, date').group('id'):
                self.assertTrue(same(rs1['pn_cnsmr, pn_manuf, pn_hi_tec']))


    def test_famac(self):
        fit = self.indport.famac('other ~ cnsmr + manuf + hi_tec + hlth')
        self.assertEqual(round(fit[0].intercept, 2), 0.02)
        self.assertEqual(round(fit[0].cnsmr, 2), 0.44)
        self.assertEqual(round(fit[0].manuf, 2), 0.16)
        self.assertEqual(round(fit[0].hi_tec, 2), 0.03)
        self.assertEqual(round(fit[0].hlth, 2), 0.10)

        fitavg = fit.tsavg()
        for var, val in zip(['cnsmr', 'manuf', 'hi_tec', 'hlth'], fitavg.lines[1][2:]):
            self.assertEqual(mean1(fit[var]), val)

    def test_rollover(self):
        lengths = []
        for rs0 in self.rs1.roll(3, 2):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [3, 3, 3, 3, 2])

        lengths = []
        for rs0 in self.rs2.where(lambda r: r.date > 200103).roll(12, 12):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [12, 12, 9])

        lengths = []
        for rs0 in self.rs2.where(lambda r: r.date > 200103).roll(24, 12):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [24, 21, 9])

        lengths = []
        for rs0 in self.rs3.roll(14, 7):
            lengths.append(len(rs0))
        self.assertEqual(lengths, [14, 14, 14, 9, 2])

        # should be able to handle missing dates
        rs = Rows([Row(date=ymd(200101, i)) for i in range(10)])
        del rs[3]
        ls = [rs1['date'] for rs1 in rs.roll(5, 4)]
        self.assertEqual(
            ls, [[200101, 200102, 200103, 200105],
                 [200105, 200106, 200107, 200108, 200109],
                 [200109, 200110]])


class TestReal(unittest.TestCase):
    def setUp(self):
        def dateid(r):
            r.date = str(r.date)[:6]
            r.id = r.fcode
            del r.fcode
            return r

        setenv(encoding='cp949')
        with dbopen('space.db') as c:
            c.save('indcode1.csv', fn=dateid)
            c.save('mdata1.csv', fn=dateid)
            c.save('manal1.csv', fn=dateid)
            c.save("mdata1 where id='A005930'" , 'sam1')
        setenv(encoding='utf-8')
    # indcode1: date, mkt, fname, icode, yyyymm, id
    # mdata1: date, ret, size, tvol, equity, pref, yyyymm, id
    # manal1, date, nfollowers, yyyymm, id
    def test_simple(self):
        with dbopen('space.db') as c:
            stmt = c.join("""
            indcode1=> date as date, id;
            mdata1:1 as md => ret * 100 as ret100, equity - pref as book;
            manal1 as an => nfollowers;
            sam1:-3 => ret as sret, tvol as stvol, an.nfollowers * md.ret as nf;
            inner join sam1:1 => ret as rets;
            """,  where='isnum(stvol)', exec=False)
            print(stmt)

            c.save(stmt, 'foo')

            c.show('foo', n=1)

            c.rename('sam1', 'dataset')

            stmt = c.join("""
            indcode1 => date, id, dx.date + date as dd;
            dataset as dx=> date as d1, id as i1, ret;
            dataset:-1 => ret as ret1, xx.ret + ret as fooo;
            dataset:-2 as xx => ret as ret2;
            dataset:-3 => ret as ret3;
            dataset:-4 => ret as ret4;
            """, 'dset0', where="dx.id='A005930'", exec=False)
            c.save(stmt, 'dset0')
            print(stmt)
            c.show('dset0', n=1)
            r0 = next(c.reel('dset0'))

            xs = c.rows('dataset where id="A005930"')['ret'][0:5]
            ys = [r0['ret']] + [r0['ret' + str(i)] for i in range(1, 5)]
            self.assertEqual(xs, ys)

            stmt = c.join("""
            indcode1;
            dataset;
            dataset:-3 => ret as ret3;
            dataset:-2 => ret as ret2;
            dataset:-1 => ret as ret1;
            dataset:-4 => ret as ret4;
            """, 'dset0', where="t0.id='A005930'", exec=False)
            print(stmt)

            c.save(stmt, 'dset0')
            r0 = next(c.reel('dset0'))

            xs = c.rows('dataset where id="A005930"')['ret'][0:5]
            ys = [r0['ret']] + [r0['ret' + str(i)] for i in range(1, 5)]
            self.assertEqual(xs, ys)


            c.save('select *, pref as reta, equity as exvwret from dataset', 'dataset')
            c.save('select date, id, ret, size from dataset', 'dset0')
            # c.show('dataset')
            c.join("""
            dataset;
            dset0:-1 => ret as ret1, size as size1;
            """, 'dset1')

            stmt = c.join("""
            dataset;
            dataset:1 => size as size1p;
            """,  exec=False, inc='inc')
            print(stmt)


    def tearDown(self):
        os.remove(os.path.join('data', 'space.db'))


class TestRegister(unittest.TestCase):
    def test_simple(self):
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

            c.run("create table test(i, j, x)")
            c.run("insert into test values (1, 3,'a')")
            c.run("insert into test values (21, 2, 'b')")
            c.run("insert into test values (5,3, 'a')")
            c.run("insert into test values (20,4, 'a')")
            c.run("insert into test values (20,'x', 'c')")
            c.run("insert into test values (20,-1.2, 'd')")

            self.assertEqual(c.rows("select foo(i, j) as val from test")['val'],
                [4, 23, 8, 24, '', 18.8])
            self.assertEqual(c.rows("select bar(i, j) as val from test")['val'],
                [4, 23, 8, 24, '', 18.8])
            self.assertEqual(
                c.rows("select foo1(i, j) as val from test group by x")['val'],
                [98, 42, '', -24.0])
            self.assertEqual(
                c.rows("select bar1(i, j) as val from test group by x")['val'],
                [98, 42, '', -24.0])


class TestMiscN(unittest.TestCase):
    def test_simple(self):
        with dbopen(':memory:') as c:
            def fn(r):
                r.date = r.date.strftime('%Y%m')
                return r
            c.save(sas('ff5_ew_mine.sas7bdat'), 'ff5', fn=fn)

            results = []
            for rs in c.roll('ff5', 11, 8, where='date >= 201501'):
                results.append(rs['date'])
            self.assertEqual(results[0], [ymd(201501, i) for i in range(11)])
            self.assertEqual(results[1], [ymd(201501, 8 + i) for i in range(5)])

            iris = c.rows('iris.csv')
            iris.show()
            c.show('iris.csv')

    def test_simple1(self):
        # empty rows
        rs = Rows([])
        rs.pn('foo', 3)
        for x in rs.roll(3, 3):
            print(x)

        for x in list(rs.group('eh')):
            print(x)


    def test_simple2(self):
        from itertools import groupby
        with dbopen(':memory:') as c:
            # if you want some performance boost
            c.save('iris.csv')
            seq = c.run('select * from iris')
            cols = [c[0] for c in seq.description]
            self.assertEqual(cols, ['col', 'sepal_length', 'sepal_width',
                                    'petal_length', 'petal_width', 'species'])
            species = []
            for i, r in groupby(seq, lambda x: x[5]):
                species.append(next(r)[5])
            self.assertEqual(species, ['setosa', 'versicolor', 'virginica'])


            xss = [(1,2,3), (2,3,4)]
            c.save(xss,'hello', cols='a,b,c')
            xss1 = []
            for r in c.run('select * from hello'):
                xss1.append(r)
            self.assertEqual(xss, xss1)

    def test_remdup(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            c.remdup('iris', cols='species')
            self.assertEqual(len(c.rows('iris')), 3)

    def test_xxx(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv')
            q = "select t1.col from iris t1"
            print(c.cols(q))

        pass


class TestSample(unittest.TestCase):
    def test_sample(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv', pkeys='col, species')
            cur = c.run('pragma table_info(iris)')
            self.assertEqual(['col', 'species'], 
                             [name for (_, name, _, _, _, pk) in cur.fetchall() if pk])

            c.save("""
            select * from iris
            """, 'iris', pkeys='sepal_length, col')
            
            cur = c.run('pragma table_info(iris)')
            # pkeys order must be preserved 
            self.assertEqual([('col', 2), ('sepal_length', 1)], 
                             [(name, pk) for (_, name, _, _, _, pk) in cur.fetchall() if pk])



    def test_sample1(self):
        with dbopen(':memory:') as c:
            c.save('iris.csv',pkeys=['col', 'col'])
            for r in c.run("pragma table_info('iris')"):
                print(r)

            for r in c.reel('select sql from sqlite_master where type="table"'):
                print(r)








# 'if __name__ ..is required
# pmap doesn't work on Windows with this thingy
if __name__ == "__main__":
    unittest.main()



