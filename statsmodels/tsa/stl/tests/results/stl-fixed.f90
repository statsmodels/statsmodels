! fixed partition sort July 2019
! Converted to double July 2019
! Converted to FORTRAN90
! Otherwise unmodified from original
subroutine ma(x, n, len, ave)
    integer n, len, i, j, k, m, newn
    real(kind=8) x(n), ave(n), flen, v
    newn = n - len + 1
    flen = dble(len)
    v = dble(0.0)
    do i = 1, len
        v = v + x(i)
    end do
    ave(1) = v / flen
    if(.not.(newn > 1))goto 23085
    k = len
    m = 0
    do j = 2, newn
        k = k + 1
        m = m + 1
        v = v - x(m) + x(k)
        ave(j) = v / flen
    end do
    23085 continue
    return
end

subroutine ess(y, n, len, ideg, njump, userw, rw, ys, res)
    integer n, len, ideg, njump, newnj, nleft, nright, nsh, k, i, j
    real(kind=8) y(n), rw(n), ys(n), res(n), delta
    logical ok, userw
    if(.not.(n < 2))goto 23019
    ys(1) = y(1)
    return
    23019 continue
    newnj = min0(njump, n - 1)
    if(.not.(len >= n))goto 23021
    nleft = 1
    nright = n
    do i = 1, n, newnj
        call est(y, n, len, ideg, dble(i), ys(i), nleft, nright, res, userw, rw, ok)
        if(.not.(.not. ok))goto 23025
        ys(i) = y(i)
        23025 continue
    end do
    goto 23022
    23021 continue
    if(.not.(newnj == 1))goto 23027
    nsh = (len + 1) / 2
    nleft = 1
    nright = len
    do i = 1, n
        if(.not.(i > nsh  .and.  nright /= n))goto 23031
        nleft = nleft + 1
        nright = nright + 1
        23031 continue
        call est(y, n, len, ideg, dble(i), ys(i), nleft, nright, res, userw, rw, ok)
        if(.not.(.not. ok))goto 23033
        ys(i) = y(i)
        23033 continue
    end do
    goto 23028
    23027 continue
    nsh = (len + 1) / 2
    do i = 1, n, newnj
        if(.not.(i < nsh))goto 23037
        nleft = 1
        nright = len
        goto 23038
        23037 continue
        if(.not.(i >= n - nsh + 1))goto 23039
        nleft = n - len + 1
        nright = n
        goto 23040
        23039 continue
        nleft = i - nsh + 1
        nright = len + i - nsh
        23040 continue
        23038 continue
        call est(y, n, len, ideg, dble(i), ys(i), nleft, nright, res, userw, rw, ok)
        if(.not.(.not. ok))goto 23041
        ys(i) = y(i)
        23041 continue
    end do
    23028 continue
    23022 continue
    if(.not.(newnj /= 1))goto 23043
    do i = 1, n - newnj, newnj
        delta = (ys(i + newnj) - ys(i)) / dble(newnj)
        do j = i + 1, i + newnj - 1
            ys(j) = ys(i) + delta * dble(j - i)
        end do
    end do
    k = ((n - 1) / newnj) * newnj + 1
    if(.not.(k /= n))goto 23049
    call est(y, n, len, ideg, dble(n), ys(n), nleft, nright, res, userw, rw, ok)
    if(.not.(.not. ok))goto 23051
    ys(n) = y(n)
    23051 continue
    if(.not.(k /= n - 1))goto 23053
    delta = (ys(n) - ys(k)) / dble(n - k)
    do j = k + 1, n - 1
        ys(j) = ys(k) + delta * dble(j - k)
    end do
    23053 continue
    23049 continue
    23043 continue
    return
end


subroutine est(y, n, len, ideg, xs, ys, nleft, nright, w, userw, rw, ok)
    integer n, len, ideg, nleft, nright, j
    real(kind=8) y(n), w(n), rw(n), xs, ys, range, h, h1, h9, a, b, c, r
    logical userw, ok
    range = dble(n) - dble(1)
    h = amax1(xs - dble(nleft), dble(nright) - xs)
    if(.not.(len > n))goto 23057
    h = h + dble((len - n) / 2)
    23057 continue
    h9 = .999 * h
    h1 = .001 * h
    a = 0.0
    do j = nleft, nright
        w(j) = 0.
        r = abs(dble(j) - xs)
        if(.not.(r <= h9))goto 23061
        if(.not.(r <= h1))goto 23063
        w(j) = 1.
        goto 23064
        23063 continue
        w(j) = (1.0 - (r / h)**3)**3
        23064 continue
        if(.not.(userw))goto 23065
        w(j) = rw(j) * w(j)
        23065 continue
        a = a + w(j)
        23061 continue
    end do
    if(.not.(a <= 0.0))goto 23067
    ok = .false.
    goto 23068
    23067 continue
    ok = .true.
    do j = nleft, nright
        w(j) = w(j) / a
    end do
    if(.not.((h > 0.) .and. (ideg > 0)))goto 23071
    a = 0.0
    do j = nleft, nright
        a = a + w(j) * dble(j)
    end do
    b = xs - a
    c = 0.0
    do j = nleft, nright
        c = c + w(j) * (dble(j) - a)**2
    end do
    if(.not.(sqrt(c) > .001 * range))goto 23077
    b = b / c
    do j = nleft, nright
        w(j) = w(j) * (b * (dble(j) - a) + 1.0)
    end do
    23077 continue
    23071 continue
    ys = 0.0
    do j = nleft, nright
        ys = ys + w(j) * y(j)
    end do
    23068 continue
    return
end

subroutine fts(x, n, np, trend, work)
    integer n, np
    real(kind=8) x(n), trend(n), work(n)
    call ma(x, n, np, trend)
    call ma(trend, n - np + 1, np, work)
    call ma(work, n - 2 * np + 2, 3, trend)
    return
end

subroutine rwts(y, n, fit, rw)
    integer mid(2), n
    real(kind=8) y(n), fit(n), rw(n), cmad, c9, c1, r
    do i = 1, n
        rw(i) = abs(y(i) - fit(i))
    end do
    ! TODO: psort does not work correctly, reorder mid
    mid(2) = n / 2 + 1
    mid(1) = n - mid(2) + 1
    call psort(rw, n, mid, 2)
    cmad = 3.0 * (rw(mid(1)) + rw(mid(2)))
    c9 = .999 * cmad
    c1 = .001 * cmad
    do i = 1, n
        r = abs(y(i) - fit(i))
        if(.not.(r <= c1))goto 23101
        rw(i) = 1.
        goto 23102
        23101 continue
        if(.not.(r <= c9))goto 23103
        rw(i) = (1.0 - (r / cmad)**2)**2
        goto 23104
        23103 continue
        rw(i) = 0.
        23104 continue
        23102 continue
    end do
    return
end


subroutine ss(y, n, np, ns, isdeg, nsjump, userw, rw, season, work1, work2, work3, work4)
    integer n, np, ns, isdeg, nsjump, nright, nleft, i, j, k
    real(kind=8) y(n), rw(n), season(n + 2 * np), work1(n), work2(n), work3(n), work4(n), xs
    logical userw, ok
    j = 1
    23105 if(.not.(j <= np))goto 23107
    k = (n - j) / np + 1
    do i = 1, k
        work1(i) = y((i - 1) * np + j)
    end do
    if(.not.(userw))goto 23110
    do i = 1, k
        work3(i) = rw((i - 1) * np + j)
    end do
    23110 continue
    call ess(work1, k, ns, isdeg, nsjump, userw, work3, work2(2), work4)
    xs = 0
    nright = min0(ns, k)
    call est(work1, k, ns, isdeg, xs, work2(1), 1, nright, work4, userw, work3, ok)
    if(.not.(.not. ok))goto 23114
    work2(1) = work2(2)
    23114 continue
    xs = k + 1
    nleft = max0(1, k - ns + 1)
    call est(work1, k, ns, isdeg, xs, work2(k + 2), nleft, k, work4, userw, work3, ok)
    if(.not.(.not. ok))goto 23116
    work2(k + 2) = work2(k + 1)
    23116 continue
    do m = 1, k + 2
        season((m - 1) * np + j) = work2(m)
    end do
    j = j + 1
    goto 23105
    23107 continue
    return
end

subroutine onestp(y, n, np, ns, nt,nl, isdeg, itdeg, ildeg, nsjump,ntjump, nljump, ni, userw,rw, season, trend, work)
    integer n, ni, np, ns, nt, nsjump, ntjump, nl, nljump, isdeg, itdeg, ildeg
    real(kind=8) y(n), rw(n), season(n), trend(n), work(n + 2 * np, 5)
    logical userw
    do j = 1, ni
        do i = 1, n
            work(i, 1) = y(i) - trend(i)
        end do
        call ss(work(1, 1), n, np, ns, isdeg, nsjump, userw, rw, work(1, 2), work(1, 3), work(1, 4), work(1, 5), season)
        call fts(work(1, 2), n + 2 * np, np, work(1, 3), work(1, 1))
        call ess(work(1, 3), n, nl, ildeg, nljump, .false., work(1, 4), work(1, 1), work(1, 5))
        do i = 1, n
            season(i) = work(np + i, 2) - work(i, 1)
        end do
        do i = 1, n
            work(i, 1) = y(i) - season(i)
        end do
        call ess(work(1, 1), n, nt, itdeg, ntjump, userw, rw, trend, work(1, 3))
    end do
    return
end

subroutine stl(y, n, np, ns, nt,nl, isdeg, itdeg, ildeg, nsjump,ntjump, nljump, ni, no,rw, season, trend, work)
    integer n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump,  nljump, ni, no, k
    integer newns, newnt, newnl, newnp
    real(kind=8) y(n), rw(n), season(n), trend(n), work(n + 2 * np, 5)
    logical userw
    userw = .false.
    k = 0
    do i = 1, n
        trend(i) = 0.0
    end do
    newns = max0(3, ns)
    newnt = max0(3, nt)
    newnl = max0(3, nl)
    newnp = max0(2, np)
    if(.not.(mod(newns, 2) == 0))goto 23002
    newns = newns + 1
    23002 continue
    if(.not.(mod(newnt, 2) == 0))goto 23004
    newnt = newnt + 1
    23004 continue
    if(.not.(mod(newnl, 2) == 0))goto 23006
    newnl = newnl + 1
    23006 continue
    23008 continue
    call onestp(y, n, newnp, newns, newnt, newnl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, userw, rw,season, trend, work)
    k = k + 1
    if(.not.(k > no))goto 23011
    goto 23010
    23011 continue
    do i = 1, n
        work(i, 1) = trend(i) + season(i)
    end do
    call rwts(y, n, work(1, 1), rw)
    userw = .true.
    23009 goto 23008
    23010 continue
    if(.not.(no <= 0))goto 23015
    do i = 1, n
        rw(i) = 1.0
    end do
    23015 continue
    return
end


subroutine stlez(y, n, np, ns, isdeg, itdeg, robust, no, rw, season, trend, work)
    logical robust
    integer n, i, j, np, ns, no, nt, nl, ni, nsjump, ntjump, nljump, newns, newnp
    integer isdeg, itdeg, ildeg
    real(kind=8) y(n), rw(n), season(n), trend(n), work(n + 2 * np, 7)
    real(kind=8) maxs, mins, maxt, mint, maxds, maxdt, difs, dift
    ildeg = itdeg
    newns = max0(3, ns)
    if(.not.(mod(newns, 2) == 0))goto 23120
    newns = newns + 1
    23120 continue
    newnp = max0(2, np)
    nt = (1.5 * newnp) / (1 - 1.5 / newns) + 0.5
    nt = max0(3, nt)
    if(.not.(mod(nt, 2) == 0))goto 23122
    nt = nt + 1
    23122 continue
    nl = newnp
    if(.not.(mod(nl, 2) == 0))goto 23124
    nl = nl + 1
    23124 continue
    if(.not.(robust))goto 23126
    ni = 1
    goto 23127
    23126 continue
    ni = 2
    23127 continue
    nsjump = max0(1, int(dble(newns) / 10 + 0.9))
    ntjump = max0(1, int(dble(nt) / 10 + 0.9))
    nljump = max0(1, int(dble(nl) / 10 + 0.9))
    do i = 1, n
        trend(i) = 0.0
    end do
    call onestp(y, n, newnp, newns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, .false., rw, season, trend, work)
    no = 0
    if(.not.(robust))goto 23130
    j = 1
    23132 if(.not.(j <= 15))goto 23134
    do i = 1, n
        work(i, 6) = season(i)
        work(i, 7) = trend(i)
        work(i, 1) = trend(i) + season(i)
    end do
    call rwts(y, n, work(1, 1), rw)
    call onestp(y, n, newnp, newns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, nljump, ni, .true., rw, season, trend, work)
    no = no + 1
    maxs = work(1, 6)
    mins = work(1, 6)
    maxt = work(1, 7)
    mint = work(1, 7)
    maxds = abs(work(1, 6) - season(1))
    maxdt = abs(work(1, 7) - trend(1))
    do i = 2, n
        if(.not.(maxs < work(i, 6)))goto 23139
        maxs = work(i, 6)
        23139 continue
        if(.not.(maxt < work(i, 7)))goto 23141
        maxt = work(i, 7)
        23141 continue
        if(.not.(mins > work(i, 6)))goto 23143
        mins = work(i, 6)
        23143 continue
        if(.not.(mint > work(i, 7)))goto 23145
        mint = work(i, 7)
        23145 continue
        difs = abs(work(i, 6) - season(i))
        dift = abs(work(i, 7) - trend(i))
        if(.not.(maxds < difs))goto 23147
        maxds = difs
        23147 continue
        if(.not.(maxdt < dift))goto 23149
        maxdt = dift
        23149 continue
    end do
    if(.not.((maxds / (maxs - mins) < .01)  .and.  (maxdt / (maxt - mint) < .01)))goto 23151
    goto 23134
    23151 continue
    j = j + 1
    goto 23132
    23134 continue
    23130 continue
    if(.not.(.not. robust))goto 23153
    do i = 1, n
        rw(i) = 1.0
    end do
    23153 continue
    return
end


subroutine psort(a, n, ind, ni)
    real(kind=8) a(n)
    integer n, ind(ni), ni
    integer indu(16), indl(16), iu(16), il(16), p, jl, ju, i, j, m, k, ij, l
    real(kind=8) t, tt
    if(.not.(n < 0 .or. ni < 0))goto 23157
    return
    23157 continue
    if(.not.(n < 2  .or.  ni == 0))goto 23159
    return
    23159 continue
    jl = 1
    ju = ni
    indl(1) = 1
    indu(1) = ni
    i = 1
    j = n
    m = 1
    23161 continue
    if(.not.(i < j))goto 23164
    go to 10
    23164 continue
    23166 continue
    m = m - 1
    if(.not.(m == 0))goto 23169
    goto 23163
    23169 continue
    i = il(m)
    j = iu(m)
    jl = indl(m)
    ju = indu(m)
    if(.not.(jl <= ju))goto 23171
    23173 if(.not.(j - i > 10))goto 23174
    10    k = i
    ij = (i + j) / 2
    t = a(ij)
    if(.not.(a(i) > t))goto 23175
    a(ij) = a(i)
    a(i) = t
    t = a(ij)
    23175 continue
    l = j
    if(.not.(a(j) < t))goto 23177
    a(ij) = a(j)
    a(j) = t
    t = a(ij)
    if(.not.(a(i) > t))goto 23179
    a(ij) = a(i)
    a(i) = t
    t = a(ij)
    23179 continue
    23177 continue
    23181 continue
    l = l - 1
    if(.not.(a(l) <= t))goto 23184
    tt = a(l)
    23186 continue
    k = k + 1
    23187 if(.not.(a(k) >= t))goto 23186
    if(.not.(k > l))goto 23189
    goto 23183
    23189 continue
    a(l) = a(k)
    a(k) = tt
    23184 continue
    23182 goto 23181
    23183 continue
    indl(m) = jl
    indu(m) = ju
    p = m
    m = m + 1
    if(.not.(l - i <= j - k))goto 23191
    il(p) = k
    iu(p) = j
    j = l
    23193 continue
    if(.not.(jl > ju))goto 23196
    goto 23167
    23196 continue
    if(.not.(ind(ju) <= j))goto 23198
    goto 23195
    23198 continue
    ju = ju - 1
    23194 goto 23193
    23195 continue
    indl(p) = ju + 1
    goto 23192
    23191 continue
    il(p) = i
    iu(p) = l
    i = k
    23200 continue
    if(.not.(jl > ju))goto 23203
    goto 23167
    23203 continue
    if(.not.(ind(jl) >= i))goto 23205
    goto 23202
    23205 continue
    jl = jl + 1
    23201 goto 23200
    23202 continue
    indu(p) = jl - 1
    23192 continue
    goto 23173
    23174 continue
    if(.not.(i == 1))goto 23207
    goto 23168
    23207 continue
    i = i - 1
    23209 continue
    i = i + 1
    if(.not.(i == j))goto 23212
    goto 23211
    23212 continue
    t = a(i + 1)
    if(.not.(a(i) > t))goto 23214
    k = i
    23216 continue
    a(k + 1) = a(k)
    k = k - 1
    23217 if(.not.(t >= a(k)))goto 23216
    a(k + 1) = t
    23214 continue
    23210 goto 23209
    23211 continue
    23171 continue
    23167 goto 23166
    23168 continue
    23162 goto 23161
    23163 continue
    return
end
