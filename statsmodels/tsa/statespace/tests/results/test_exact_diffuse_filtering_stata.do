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
// Have to skip this since Stata does not allow missing values, even using sspace
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

// Dynamic Factor Model
clear
input y1 y2 t
9.9768523266  6.1144429663 1
-0.4771808443  4.1543910949 2
1.3978130617  0.4336043785 3
8.8760718057  3.8136603507 4
-1.8738213131  5.0289712198 5
0.6531520756 -1.5871706073 6
-5.1625439784  0.5372132872 7
2.3690365022 -0.1118599521 8
7.4138136616  5.9079363820 9
6.4126556647  1.9354521733 10
8.0611264188  7.9292248028 11
7.1109037147  4.2364664530 12 
4.3922061400  4.8864935148 13
3.6816268854  3.2248106820 14
0.9708074857  5.6330208687 15
5.1940841814  2.6849177229 16
4.9811338373  3.8017109128 17
7.4616162956  5.4061666493 18
3.0295447157  3.3396476674 19
8.8783455475  7.8216699141 20
end
tsset t

constraint 1 [u2]L.u1 = 1

matrix define b0 = (0.9, 0.1, 1, 0.5, 1, 1, 1.5, 2.)
sspace (u1 L.u1 L.u2, state noconstant) ///
	   (u2 L.u1, state noconstant noerror) ///
       (y1 u1, noconstant) ///
       (y2 u1, noconstant), ///
       constraints(1) covstate(diagonal) ///
       from(b0) iterate(0)
