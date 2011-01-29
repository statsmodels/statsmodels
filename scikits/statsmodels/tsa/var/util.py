from cStringIO import StringIO

import numpy as np

def parse_data(path):
    """
    Parse data files from Lutkepohl (2005) book

    Source for data files: www.jmulti.de
    """

    from collections import deque
    from datetime import datetime
    import pandas
    import pandas.core.datetools as dt
    import re

    regex = re.compile('<(.*) (\w)([\d]+)>.*')
    lines = deque(open(path))

    to_skip = 0
    while '*/' not in lines.popleft():
        to_skip += 1

    while True:
        to_skip += 1
        line = lines.popleft()
        m = regex.match(line)
        if m:
            year, freq, start_point = m.groups()
            break

    data = np.genfromtxt(path, names=True, skip_header=to_skip+1)

    n = len(data)

    # generate the corresponding date range (using pandas for now)
    start_point = int(start_point)
    year = int(year)

    offsets = {
        'Q' : dt.BQuarterEnd(),
        'M' : dt.BMonthEnd(),
        'A' : dt.BYearEnd()
    }

    # create an instance
    offset = offsets[freq]

    inc = offset * (start_point - 1)
    start_date = offset.rollforward(datetime(year, 1, 1)) + inc

    offset = offsets[freq]
    date_range = pandas.DateRange(start_date, offset=offset, periods=n)

    return data, date_range


def pprint_matrix(values, rlabels, clabels, col_space=None):
    buf = StringIO()

    T, K = len(rlabels), len(clabels)

    if col_space is None:
        min_space = 10
        col_space = [max(len(str(c)) + 2, min_space) for c in clabels]
    else:
        col_space = (col_space,) * K

    row_space = max([len(str(x)) for x in rlabels]) + 2

    head = _pfixed('', row_space)

    for j, h in enumerate(clabels):
        head += _pfixed(h, col_space[j])

    print >> buf, head

    for i, rlab in enumerate(rlabels):
        line = ('%s' % rlab).ljust(row_space)

        for j in range(K):
            line += _pfixed(values[i,j], col_space[j])

        print >> buf, line

    return buf.getvalue()

def _pfixed(s, space, nanRep=None, float_format=None):
    if isinstance(s, float):
        if float_format:
            formatted = float_format(s)
        else:
            formatted = "%#8.6F" % s

        return formatted.rjust(space)
    else:
        return ('%s' % s)[:space].rjust(space)
