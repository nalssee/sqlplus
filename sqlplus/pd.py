"""
"""

from itertools import product, groupby

from sqlplus.core import Rows, Row
from sqlplus.util import chunks, isnum

# {'a': '3,2,3',
#  'b': 'b < 0, b >=0 and b <= 100: 3 4 5, b > 100']}

# {'a': 3,
#  'b': [3, 4, 3],
#  'c':
#  'd': [(-inf, 0 - epsilon), (0, 10, 3), (10+epsilon, inf)]


class PD:
    def __init__(self, conn, tname=None, dcol=None, icol=None):
        self.conn = conn
        self.tname = tname
        self.dcol = dcol
        self.icol = icol

    def breaks(self, cksizes, dep=False):
        # cksizes: chunk sizes
        def accum(xs):
            return [xs[:i] for i in range(1, len(xs) + 1)]

        def flatten(xs):
            # flatten a list of lists
            return [x1 for x in xs for x1 in x]

        # rs is ordered first
        def breaks1(rs, sizes):
            if isinstance(sizes, int):



            if isnum(*sizes):



            # str to Rows
            sizes = [(rs.where(s) if isinstance(s, str) else s) for s in sizes]
            gsizes = [(t, list(x)) for t, x in groupby(sizes, type)]
            return [to_chunks(t, x, gsizes) for t, x in gsizes]

        def ipn(rs):
            for col, sizes in cksizes.items():
                pncol = 'pn_' + col
                for i, rs in enumerate(churs.order(col)._chunks(ps, c), 1):
                    rs[pncol] = i
            return self

        def ibreaks():
            self._ipn(cps)
            intvals = []
            for col, pncol in zip(cols, pncols):
                self.order(col)
                bps1 = [rs[-1][col] for rs in self.group(pncol)]
                intvals1 = [(a, b) for a, b in zip([float('-inf')] + bps1, bps1[:-1] + [float('+inf')])]
                intvals.append(intvals1)

            for pns, intval in zip_longest(product(*(range(1, s + 1) for s in sizes)),
                                        product(*intvals)):
                assert intval, f'Not enough obs at {date}'
                yield pns, intval

        def dpn(self, cps):
            prev_cols = []
            for c, ps in cps.items():
                pncol = 'pn_' + c
                for rs in self.group(prev_cols):
                    for i, rs1 in enumerate(rs.order(c)._chunks(ps, c), 1):
                        rs1[pncol] = i
                prev_cols.append(pncol)
            return self

        def dbreaks():
            self._dpn(cps)
            self.show(file='sample.csv')
            d = {}
            def ab(pns, rs, col, size):
                *pn0, n = pns
                if n == 1:
                    # already well ordered. so no need to use
                    return (float('-inf'), max(r[col] for r in rs))
                elif n < size:
                    return (d[(*pn0, n - 1)][-1][-1], max(r[col] for r in rs))
                else:
                    return (d[(*pn0, n - 1)][-1][-1], float('+inf'))

            for pncols1 in accum(pncols):
                for rs in self.group(pncols1):
                    r0 = rs[0]
                    pns = tuple(r0[c] for c in pncols1)
                    col = cols[len(pns) - 1]
                    size = sizes[len(pns) - 1]
                    d[pns] = d.get(pns[:-1], []) + [ab(pns, rs, col, size)]
            for pns in product(*(range(1, s + 1) for s in sizes)):
                try:
                    yield pns, d[pns]
                except:
                    raise ValueError(f'Not enough obs at {date}')

        def fn(rs):
            date = rs[0][dcol]
            result = []
            for (pns, intval) in (dbreaks() if dep else ibreaks()):
                r = Row()
                r[dcol] = date
                for pncol, pn1, (a, b) in zip(pncols, pns, intval):
                    r[pncol] = pn1
                    r[pncol + '_min'] = a
                    r[pncol + '_max'] = b
                result.append(r)
            return result

        result = []
        for rs in conn.reel(tname, group=dcol):
            result += fn(rs)

        return Rows(result)


def pavg(conn, tname, bps):
    pass

def tavg():
    pass




# dependent portfolio
# def pn(self, cps, dcol=None, icol=None, dep=False):
#     "Number portfolios based on the first date"
#     # self can't be empty
#     if isinstance(cps, dict):

#         cols = list(cps)
#         pncols = ['pn_' + c for c in cols]
#         self[pncols] = ''
#         self.order(dcol)
#         rs = next(self.group(dcol, order=False))

#         if dep:
#             rs._dpn(cps)
#         else:
#             rs._ipn(cps)
#         # python sort preserves order
#         for rs1 in self.group(icol):
#             for pncol in pncols:
#                 rs1[pncol] = rs1[0][pncol]
#         return self
#     # cps is an instance of Rows
#     pncols = [c for c in cps[0].columns if c.startswith('pn_') \
#               and not c.endswith('_max') and not c.endswith('_min')]
#     cols = [c[3:] for c in pncols]
#     self[pncols] = ''
#     self.order(dcol)
#     rs = next(self.group(dcol, order=False))

#     for r in cps.where({dcol: rs[0][dcol]}):
#         rs1 = rs.where({c: [r[c + '_min'], r[c + '_max']] for c in pncols})
#         for c in pncols:
#             rs1[c] = r[c]
#     return self


# def pavg(self, col, wcol=None, dcol=None):
#     # portfolio avg for one date
#     date = self[0][dcol] if dcol else None
#     pncols = [c for c in self[0].columns if c.startswith('pn_')]
#     rs = self.isnum(col, wcol) if wcol else self.isnum(col)
#     result = []
#     for rs1 in rs:
#         r0 = Row()
#         if dcol:
#             r0[dcol] = rs1[0][dcol]
#         for c in pncols:
#             r0[c] = rs1[0][c]
#         r0[col] = rs1.avg(col, wcol)
#         r0.nobs = len(rs1)
#         result.append(r0)
#     return self._newrows(result)


