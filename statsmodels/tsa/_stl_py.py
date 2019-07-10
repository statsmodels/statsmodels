"""Pure python port fo STL FORTRAN functions"""
from numpy import sqrt, abs as npabs, empty, partition, round


def est(y, n, len_, ideg, xs, nleft, nright, w, userw, rw):
    # Removed ok and ys, which are scalar return values
    rng = n - 1.0
    h = max(xs - nleft, nright - xs)
    if len_ > n:
        h += (len_ - n) // 2.0
    h9 = .999 * h
    h1 = .001 * h
    a = 0.0
    for j in range(nleft - 1, nright):
        w[j] = 0.
        r = abs(j + 1 - xs)
        if r <= h9:
            if r <= h1:
                w[j] = 1.0
            else:
                w[j] = (1.0 - (r / h) ** 3) ** 3
            if userw:
                w[j] = w[j] * rw[j]
            a = a + w[j]
    if a <= 0:
        ok = False
        return 0.0, ok
    ok = True
    for j in range(nleft - 1, nright):
        w[j] = w[j] / a
    if h > 0 and ideg > 0:
        a = 0.0
        for j in range(nleft - 1, nright):
            a = a + w[j] * (j + 1)
        b = xs - a
        c = 0.0
        for j in range(nleft - 1, nright):
            c = c + w[j] * (j + 1 - a) ** 2
        if sqrt(c) > .001 * rng:
            b = b / c
            for j in range(nleft - 1, nright):
                w[j] = w[j] * (b * (j + 1 - a) + 1.0)
    ys = 0.0
    for j in range(nleft - 1, nright):
        ys = ys + w[j] * y[j]

    return ys, ok


def ma(x, n, len_, ave):
    newn = n - len_ + 1
    flen = float(len_)
    v = 0.0
    for i in range(len_):
        v = v + x[i]
    ave[0] = v / flen
    if newn <= 1:
        return
    k = len_
    m = 0
    for j in range(1, newn):
        v += x[k] - x[m]
        ave[j] = v / flen
        k += 1
        m += 1


def fts(x, n, np, trend, work):
    ma(x, n, np, trend)
    ma(trend, n - np + 1, np, work)
    ma(work, n - 2 * np + 2, 3, trend)


def rwts(y, n, fit, rw):
    rw[:] = npabs(y - fit[:n])
    mid = empty(2, dtype=int)
    mid[0] = n // 2
    mid[1] = n - mid[0] - 1
    rw_part = partition(rw, mid)
    cmad = 3.0 * (rw_part[mid[0]] + rw_part[mid[1]])
    c9 = .999 * cmad
    c1 = .001 * cmad
    small = rw <= c1
    large = rw > c9
    # TODO: Could produce runtime warnings for division
    rw[:] = (1.0 - (rw / cmad) ** 2) ** 2
    # TODO: Need data that crosses these values for coverage
    if small.any():
        rw[small] = 1.0
    if large.any():
        rw[large] = 0.0


def ss(y, n, np, ns, isdeg, nsjump, userw, rw, season, work1, work2, work3,
       work4):
    # TODO: the iss in short is in here!!!
    for j in range(np):
        k = (n - (j + 1)) // np + 1  # +1 here due to indexing diff
        for i in range(k):
            work1[i] = y[i * np + j]
        if userw:
            for i in range(k):
                work3[i] = rw[i * np + j]
        ess(work1, k, ns, isdeg, nsjump, userw, work3, work2[1:], work4)
        xs = 0
        nright = min(ns, k)
        work2[0], ok = est(work1, k, ns, isdeg, xs, 1, nright, work4, userw,
                           work3)  # remove work2(1), ok
        if not ok:
            work2[0] = work2[1]
        xs = k + 1
        nleft = max(1, k - ns + 1)
        work2[k + 1], ok = est(work1, k, ns, isdeg, xs, nleft, k, work4, userw,
                               work3)
        if not ok:
            work2[k + 1] = work2[k]
        for m in range(k + 2):
            season[m * np + j] = work2[m]


def ess(y, n, len_, ideg, njump, userw, rw, ys, res):
    # TODO: Try with 1 data point!!? Establish minimums
    if n < 2:
        ys[0] = y[0]
        return
    newnj = min(njump, n - 1)
    if len_ >= n:
        nleft = 1
        nright = n
        for i in range(0, n, newnj):
            ys[i], ok = est(y, n, len_, ideg, i + 1, nleft, nright, res, userw,
                            rw)  # i+1 to correct for indexing diff
            if not ok:
                ys[i] = y[i]
    elif newnj == 1:
        nsh = (len_ + 2) // 2
        nleft = 1
        nright = len_
        for i in range(n):
            if (i + 1) > nsh and nright != n:
                nleft = nleft + 1
                nright = nright + 1
            ys[i], ok = est(y, n, len_, ideg, i + 1, nleft, nright, res, userw,
                            rw)  # i+1 to correct for indexing diff
            if not ok:
                ys[i] = y[i]
    else:
        nsh = (len_ + 1) // 2
        for i in range(0, n, newnj):
            if (i + 1) < nsh:
                nleft = 1
                nright = len_
            elif (i + 1) >= (n - nsh + 1):
                nleft = n - len_ + 1
                nright = n
            else:
                nleft = i + 1 - nsh + 1
                nright = len_ + i + 1 - nsh
            ys[i], ok = est(y, n, len_, ideg, i + 1, nleft, nright, res, userw,
                            rw)
            if not ok:
                ys[i] = y[i]
    if newnj != 1:  # TODO: Invert and return early
        for i in range(0, n - newnj, newnj):
            delta = (ys[i + newnj] - ys[i]) / newnj
            for j in range(i, i + newnj):
                ys[j] = ys[i] + delta * ((j + 1) - (i + 1))
        k = ((n - 1) // newnj) * newnj + 1
        if k != n:
            ys[n - 1], ok = est(y, n, len_, ideg, n, nleft, nright, res, userw,
                                rw)
            if not ok:
                ys[n - 1] = y[n - 1]
            if k != (n - 1):
                delta = (ys[n - 1] - ys[k - 1]) / (n - k)
                for j in range(k, n):
                    ys[j] = ys[k - 1] + delta * ((j + 1) - k)


def onestp(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump,
           ni, userw, rw, season, trend, work):
    for j in range(ni):
        for i in range(n):
            work[i, 0] = y[i] - trend[i]
        ss(work[:, 0], n, np, ns, isdeg, nsjump, userw, rw, work[:, 1],
           work[:, 2], work[:, 3], work[:, 4], season)
        fts(work[:, 1], n + 2 * np, np, work[:, 2], work[:, 0])
        ess(work[:, 2], n, nl, ildeg, nljump, False, work[:, 3], work[:, 0],
            work[:, 4])
        for i in range(n):
            season[i] = work[np + i, 1] - work[i, 0]
        for i in range(n):
            work[i, 0] = y[i] - season[i]
        ess(work[:, 0], n, nt, itdeg, ntjump, userw, rw, trend, work[:, 2])


def stl(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni,
        no, rw, season, trend, work):
    userw = False
    k = 0
    trend[:] = 0
    newns = max(3, ns)
    newnt = max(3, nt)
    newnl = max(3, nl)
    newnp = max(2, np)
    if (newns % 2) == 0:
        newns = newns + 1  # make odd
    if (newnt % 2) == 0:
        newnt = newnt + 1
    if (newnl % 2) == 0:
        newnl = newnl + 1
    while True:
        onestp(y, n, newnp, newns, newnt, newnl, isdeg, itdeg, ildeg, nsjump,
               ntjump, nljump, ni, userw, rw, season, trend, work)
        k = k + 1
        if k > no:
            break
        for i in range(n):
            work[i, 0] = trend[i] + season[i]
        rwts(y, n, work[:, 0], rw)
        userw = True
    if no <= 0:
        rw[:] = 1.0

    return


def stlez(y, n, np, ns, isdeg, itdeg, robust, rw, season, trend, work):
    ildeg = itdeg
    newns = max(3, ns)
    newns = newns + 1 if newns % 2 == 0 else newns
    newnp = max(2, np)
    nt = int(round(int(1.5 * newnp) // (1 - int(1.5 / newns)) + 0.5))
    nt = max(3, nt)
    nt = nt + 1 if (nt % 2) == 0 else nt
    nl = newnp
    nl = nl + 1 if nl % 2 == 0 else nl
    ni = 1 if robust else 2
    nsjump = max(1, int(newns / 10. + 0.9))
    ntjump = max(1, int(nt / 10. + 0.9))
    nljump = max(1, int(nl / 10. + 0.9))
    trend[:] = 0
    onestp(y, n, newnp, newns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump,
           nljump, ni, False, rw, season, trend, work)
    no = 0
    if not robust:
        rw[:] = 1.0
        return no
    for j in range(15):
        for i in range(n):
            work[i, 5] = season[i]
            work[i, 6] = trend[i]
            work[i, 0] = trend[i] + season[i]
        rwts(y, n, work[:, 0], rw)
        onestp(y, n, newnp, newns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump,
               nljump, ni, True, rw, season, trend, work)
        no = no + 1
        maxs = work[0, 5]
        mins = work[0, 5]
        maxt = work[0, 6]
        mint = work[0, 6]
        maxds = abs(work[0, 5] - season[0])
        maxdt = abs(work[0, 6] - trend[0])
        for i in range(1, n):
            if maxs < work[i, 5]:
                maxs = work[i, 5]
            if maxt < work[i, 6]:
                maxt = work[i, 6]
            if mins > work[i, 5]:
                mins = work[i, 5]
            if mint > work[i, 6]:
                mint = work[i, 6]
            difs = abs(work[i, 5] - season[i])
            dift = abs(work[i, 6] - trend[i])
            if maxds < difs:
                maxds = difs
            if maxdt < dift:
                maxdt = dift
        if maxds / (maxs - mins) < .01 and maxdt / (maxt - mint) < .01:
            break
    return no
