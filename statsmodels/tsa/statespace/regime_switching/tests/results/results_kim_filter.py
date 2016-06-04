'''
Data for Kim filter and switching MLE model testing.
Copied from http://econ.korea.ac.kr/~cjkim/MARKOV/data/kim_je.prn and from
http://econ.korea.ac.kr/~cjkim/MARKOV/programs/kim_je.opt optimization results.
See chapter 5.4.2 of Kim and Nelson book and for details.
'''

kim_je = {
	'data': [
		1056.5, 1063.2, 1067.1, 1080.0, 1086.8, 1106.1, 1116.3, 1125.5, 1112.4,
		1105.9, 1114.3, 1103.3, 1148.2, 1181.0, 1225.3, 1260.2, 1286.6, 1320.4,
		1349.8, 1356.0, 1369.2, 1365.9, 1378.2, 1406.8, 1431.4, 1444.9, 1438.2,
		1426.6, 1406.8, 1401.2, 1418.0, 1438.8, 1469.6, 1485.7, 1505.5, 1518.7,
		1515.7, 1522.6, 1523.7, 1540.6, 1553.3, 1552.4, 1561.5, 1537.3, 1506.1,
		1514.2, 1550.0, 1586.7, 1606.4, 1637.0, 1629.5, 1643.4, 1671.6, 1666.8,
		1668.4, 1654.1, 1671.3, 1692.1, 1716.3, 1754.9, 1777.9, 1796.4, 1813.1,
		1810.1, 1834.6, 1860.0, 1892.5, 1906.1, 1948.7, 1965.4, 1985.2, 1993.7,
		2036.9, 2066.4, 2099.3, 2147.6, 2190.1, 2195.8, 2218.3, 2229.2, 2241.8,
		2255.2, 2287.7, 2300.6, 2327.3, 2366.9, 2385.3, 2383.0, 2416.5, 2419.8,
		2433.2, 2423.5, 2408.6, 2406.5, 2435.8, 2413.8, 2478.6, 2478.4, 2491.1,
		2491.0, 2545.6, 2595.1, 2622.1, 2671.3, 2734.0, 2741.0, 2738.3, 2762.8,
		2747.4, 2755.2, 2719.3, 2695.4, 2642.7, 2669.6, 2714.9, 2752.7, 2804.4,
		2816.9, 2828.6, 2856.8, 2896.0, 2942.7, 3001.8, 2994.1, 3020.5, 3115.9,
		3142.6, 3181.6, 3181.7, 3178.7, 3207.4, 3201.3, 3233.4, 3157.0, 3159.1,
		3199.2, 3261.1, 3250.2, 3264.6, 3219.0, 3170.4, 3179.9, 3154.5, 3159.3,
		3186.6, 3258.3, 3306.4, 3365.1, 3444.7, 3487.1, 3507.4, 3520.4, 3547.0,
		3567.6, 3603.8, 3622.3, 3655.9, 3661.4, 3686.4, 3698.3
    ],
    'start': 22,
    'untransformed_start_parameters': [-3.5, 0.0, 0.5, 1.0, 0.7, -0.4, 1.7],
    'parameters': [
            1 - 0.950262, 0.442799, 1.260842, -0.353435, 0.801414, -1.291663,
            2.237430
    ],
    'loglike': -178.915776,
    'cycle': [
        4.311466, 4.767204, 4.698398, 3.708395, 2.355974, 1.032506, -0.268984,
        -0.345743, -0.030241, 0.670987, 0.912969, 1.110768, 1.083697, 0.402878,
        -0.082521, -0.668662, -0.665322, -0.698703, -1.284562, -1.619743,
        -1.931419, -2.252820, -2.477575, -1.506981, -0.407614, -0.025110,
        0.614708, 0.174542, -0.106522, 0.349356, -0.124420, -0.838504,
        -1.504355, -1.691064, -1.459829, -1.108731, -0.332804, 0.086006,
        0.189553, 0.185387, -0.407124, -0.305929, 0.018063, 0.555756, 0.506122,
        1.270411, 1.335450, 1.317515, 0.945476, 1.673555, 2.094997, 2.525956,
        3.450711, 4.210815, 3.729308, 3.618201, 3.205845, 2.803721, 2.443112,
        2.714097, 2.412336, 2.480770, 3.005782, 2.874348, 2.020085, 2.248645,
        1.640500, 1.229074, 0.211178, -0.964556, -1.911468, -1.785557,
        -2.060578, -0.429713, -0.534415, -0.994942, -1.594309, -0.816456,
        -0.022787, 0.169274, 0.753080, 1.726797, 1.440452, 0.638814, 0.448078,
        -0.368337, -1.033846, -1.422991, -1.529685, -1.141396, -1.194268,
        -0.789590, -0.406526, 0.144394, 0.055597, -0.313836, -0.361403,
        -0.101791, 0.347058, 1.054144, 0.620299, 0.373876, 1.698107, 1.896905,
        1.995808, 1.399933, 0.575056, 0.391762, -0.305998, -0.406515,
        -1.123085, -1.647202, -1.509950, -0.790480, -1.094653, -1.632443,
        -1.605995, -1.451853, -1.717052, -1.595366, -2.247132, -2.414307,
        -1.899244, -1.304171, -0.817278, -0.049496, 0.356571, 0.231505,
        -0.118546
    ]
}
