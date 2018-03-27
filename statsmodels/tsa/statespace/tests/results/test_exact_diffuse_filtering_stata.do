// Local level
clear
input x t
10.2394 1
1 2
1 3
1 4
1 5
1 6
1 7
1 8
1 9
1 10
end
tsset t

matrix define b0 = (8.253, 1.993)
ucm x, model(llevel) from(b0) iterate(0)

// e(ll) = -23.9352603142740605
disp %20.19g e(ll)

// Local linear trend
clear
input x t
10.2394 1
4.2039 2
6.123123 3
1 4
1 5
1 6
1 7
1 8
1 9
1 10
end
tsset t

matrix define b0 = (8.253, 2.334, 1.993)
ucm x, model(lltrend) from(b0) iterate(0)

// e(ll) = -22.9743755748041529
disp %20.19g e(ll)


// Local linear trend + missing
// Have to skip this since Stata doesn't allow missing values, even using sspace
// clear
// input x t
// 10.2394 1
// . 2
// 6.123123 3
// 1 4
// 1 5
// 1 6
// 1 7
// 1 8
// 1 9
// 1 10
// end
// tsset t
// 
// constraint 1 [x]u1 = 1
// constraint 2 [u1]L.u1 = 1
// constraint 3 [u1]L.u2 = 1
// constraint 4 [u2]L.u2 = 1
// 
// matrix define b0 = (1, 1, 1, 1, 8.253, 2.334, 1.993)
// sspace (u1 L.u1 L.u2, state noconstant) ///
//        (u2 L.u2, state noconstant) ///
//        (x u1, noconstant), ///
//        constraints(1/4) covstate(diagonal) ///
//        from(b0) iterate(0)
// 
// disp %20.19g e(ll)

// Common trend
clear
input y1 y2 t
10.2394 8.2304 1
1 1 2
1 1 3
1 1 4
1 1 5
1 1 6
1 1 7
1 1 8
1 1 9
1 1 10
end
tsset t

constraint 1 [y1]u1 = 1
constraint 2 [y2]u2 = 1
constraint 3 [u1]L.u1 = 1
constraint 4 [u2]L.u2 = 1

matrix define b0 = (1, 1, 1, 0.1111, 1, 3.2324, 1, 1)
sspace (u1 L.u1, state noconstant) ///
       (u2 L.u2, state noconstant noerror) ///
       (y1 u1, noconstant) ///
       (y2 u1 u2, noconstant), ///
       constraints(1/4) covstate(diagonal) ///
       from(b0) iterate(0)

// -53.7830389463984773
disp %20.19g e(ll)
