! data are 348 monthly values (29 years) in file stl_co2
! gfortran stl-fixed.f90 stl_test_data.f90 -o stl_test
! ./stl_test < co2 > stl_test_results.csv
real(kind=8) y(348), season(348), trend(348), rw(348), work(372, 7), &
             y_short(347), season_short(347), trend_short(347), rw_short(347), &
             work_short(371, 7)
logical robust, ok, userw
read(5, *)(y(i), i = 1, 348)

print *, "scenario,idx,season,trend,rw"

n = 348
np = 12
ns = 35
nt = 19
nl = 13
no = 2
ni = 1
nsjump = 4
ntjump = 2
nljump = 2
isdeg = 1
itdeg = 1
ildeg = 1
trend = 0.0
season = 0.0
rw = 1.0
work = 0.0
call stl(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, &
         nljump, ni, no, rw, season, trend, work)
do i = 1, n
    print *, "baseline,", i-1, ",", season(i), ",", trend(i), ",", rw(i)
end do

nljump = 1
trend = 0.0
season = 0.0
rw = 1.0
work = 0.0
call stl(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, &
         nljump, ni, no, rw, season, trend, work)
do i = 1, n
    print *, "nljump-1,", i-1, ",", season(i), ",", trend(i), ",", rw(i)
end do

nljump = 2
ntjump = 1
trend = 0.0
season = 0.0
rw = 1.0
work = 0.0
call stl(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, &
         nljump, ni, no, rw, season, trend, work)
do i = 1, n
    print *, "ntjump-1,", i-1, ",", season(i), ",", trend(i), ",", rw(i)
end do

rw = 1.0
nljump = 1
ntjump = 1
trend = 0.0
season = 0.0
rw = 1.0
work = 0.0
call stl(y, n, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, &
         nljump, ni, no, rw, season, trend, work)
do i = 1, n
    print *, "nljump-1-ntjump-1,", i-1, ",", season(i), ",", trend(i), ",", rw(i)
end do


do i = 1, n-1
    y_short(i) = y(i)
end do
ntjump = 2
nljump = 2
rw_short = 1.0
trend_short = 0.0
season_short = 0.0
work_short = 0.0
call stl(y_short, n-1, np, ns, nt, nl, isdeg, itdeg, ildeg, nsjump, ntjump, &
         nljump, ni, no, rw_short, season_short, trend_short, work_short)
do i = 1, n-1
    print *, "short,", i-1, ",", season_short(i), ",", trend_short(i), ",", rw_short(i)
end do

season = 0.0
trend = 0.0
rw = 1.0
robust = .false.
call stlez(y, n, np, ns, isdeg, itdeg, robust, nconv, rw, season, trend, work)
do i = 1, n
    print *, "stlez,", i-1, ",", season(i), ",", trend(i), ",", rw(i)
end do

season = 0.0
trend = 0.0
rw = 1.0
robust = .true.
call stlez(y, n, np, ns, isdeg, itdeg, robust, nconv, rw, season, trend, work)
do i = 1, n
    print *, "stlez-robust,", i-1, ",", season(i), ",", trend(i), ",", rw(i)
end do

end
