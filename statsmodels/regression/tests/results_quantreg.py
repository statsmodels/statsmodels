import numpy as np
class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__  = self

biweight_bofinger = Bunch()
biweight_bofinger.table = np.array([
    [ .5601805 , .0136491 , 41.04 , 0.000 , .533289 , .5870719],
    [ 81.48233 , 15.1604 , 5.37 , 0.000 , 51.61335 , 111.3513]
 ])
biweight_bofinger.psrsquared     =    0.6206
biweight_bofinger.rank =  2
biweight_bofinger.sparsity =  216.8218989750115
biweight_bofinger.bwidth =  .2173486679767846
biweight_bofinger.kbwidth =  91.50878448104551
biweight_bofinger.df_m =  1
biweight_bofinger.df_r =  233
biweight_bofinger.f_r =  .0046120802590851
biweight_bofinger.N =  235
biweight_bofinger.q_v =  582.541259765625
biweight_bofinger.q =  .5
biweight_bofinger.sum_rdev =  46278.05667114258
biweight_bofinger.sum_adev =  17559.93220318131
biweight_bofinger.convcode =  0


biweight_hsheather = Bunch()
biweight_hsheather.table = np.array([
    [.5601805 , .0128449 , 43.61 , 0.000 , .5348735 , .5854875],
    [81.48233 , 14.26713 , 5.71 , 0.000 , 53.37326 , 109.5914]
 ])
biweight_hsheather.psrsquared     =    0.6206
biweight_hsheather.rank =  2
biweight_hsheather.sparsity =  204.0465407204423
biweight_hsheather.bwidth =  .1574393314202373
biweight_hsheather.kbwidth =  64.53302151153288
biweight_hsheather.df_m =  1
biweight_hsheather.df_r =  233
biweight_hsheather.f_r =  .0049008427022052
biweight_hsheather.N =  235
biweight_hsheather.q_v =  582.541259765625
biweight_hsheather.q =  .5
biweight_hsheather.sum_rdev =  46278.05667114258
biweight_hsheather.sum_adev =  17559.93220318131
biweight_hsheather.convcode =  0

biweight_chamberlain = Bunch()
biweight_chamberlain.table = np.array([
    [ .5601805 , .0114969 , 48.72 , 0.000 , .5375294 , .5828315],
    [ 81.48233 , 12.76983 , 6.38 , 0.000 , 56.32325 , 106.6414]
])
biweight_chamberlain.psrsquared     =    0.6206
biweight_chamberlain.rank =  2
biweight_chamberlain.sparsity =  182.6322495257494
biweight_chamberlain.bwidth =  .063926976464458
biweight_chamberlain.kbwidth =  25.61257055690209
biweight_chamberlain.df_m =  1
biweight_chamberlain.df_r =  233
biweight_chamberlain.f_r =  .005475484218131
biweight_chamberlain.N =  235
biweight_chamberlain.q_v =  582.541259765625
biweight_chamberlain.q =  .5
biweight_chamberlain.sum_rdev =  46278.05667114258
biweight_chamberlain.sum_adev =  17559.93220318131
biweight_chamberlain.convcode =  0

epanechnikov_bofinger = Bunch()
epanechnikov_bofinger.table = np.array([
    [ .5601805 , .0209663 , 26.72 , 0.000 , .5188727 , .6014882],
    [ 81.48233 , 23.28774 , 3.50 , 0.001 , 35.60088 , 127.3638]
    ])
epanechnikov_bofinger.psrsquared     =    0.6206
epanechnikov_bofinger.rank =  2
epanechnikov_bofinger.sparsity =  333.0579553401614
epanechnikov_bofinger.bwidth =  .2173486679767846
epanechnikov_bofinger.kbwidth =  91.50878448104551
epanechnikov_bofinger.df_m =  1
epanechnikov_bofinger.df_r =  233
epanechnikov_bofinger.f_r =  .0030024804511235
epanechnikov_bofinger.N =  235
epanechnikov_bofinger.q_v =  582.541259765625
epanechnikov_bofinger.q =  .5
epanechnikov_bofinger.sum_rdev =  46278.05667114258
epanechnikov_bofinger.sum_adev =  17559.93220318131
epanechnikov_bofinger.convcode =  0

epanechnikov_hsheather = Bunch()
epanechnikov_hsheather.table = np.array([
    [.5601805 , .0170484 , 32.86 , 0.000 , .5265918 , .5937692],
    [81.48233 , 18.93605 , 4.30 , 0.000 , 44.17457 , 118.7901]
    ])
epanechnikov_hsheather.psrsquared     =    0.6206
epanechnikov_hsheather.rank =  2
epanechnikov_hsheather.sparsity =  270.8207209067576
epanechnikov_hsheather.bwidth =  .1574393314202373
epanechnikov_hsheather.kbwidth =  64.53302151153288
epanechnikov_hsheather.df_m =  1
epanechnikov_hsheather.df_r =  233
epanechnikov_hsheather.f_r =  .0036924796472434
epanechnikov_hsheather.N =  235
epanechnikov_hsheather.q_v =  582.541259765625
epanechnikov_hsheather.q =  .5
epanechnikov_hsheather.sum_rdev =  46278.05667114258
epanechnikov_hsheather.sum_adev =  17559.93220318131
epanechnikov_hsheather.convcode =  0

epanechnikov_chamberlain = Bunch()
epanechnikov_chamberlain.table = np.array([
    [.5601805 , .0130407 , 42.96 , 0.000 , .5344876 , .5858733],
    [81.48233 , 14.48467 , 5.63 , 0.000 , 52.94468 , 110.02]
    ])
epanechnikov_chamberlain.psrsquared     =    0.6206
epanechnikov_chamberlain.rank =  2
epanechnikov_chamberlain.sparsity =  207.1576340635951
epanechnikov_chamberlain.bwidth =  .063926976464458
epanechnikov_chamberlain.kbwidth =  25.61257055690209
epanechnikov_chamberlain.df_m =  1
epanechnikov_chamberlain.df_r =  233
epanechnikov_chamberlain.f_r =  .0048272418466269
epanechnikov_chamberlain.N =  235
epanechnikov_chamberlain.q_v =  582.541259765625
epanechnikov_chamberlain.q =  .5
epanechnikov_chamberlain.sum_rdev =  46278.05667114258
epanechnikov_chamberlain.sum_adev =  17559.93220318131
epanechnikov_chamberlain.convcode =  0

epan2_bofinger = Bunch()
epan2_bofinger.table = np.array([
    [.5601805 , .0143484 , 39.04 , 0.000 , .5319113 , .5884496],
    [81.48233 , 15.93709 , 5.11 , 0.000 , 50.08313 , 112.8815]
    ])
epan2_bofinger.psrsquared     =    0.6206
epan2_bofinger.rank =  2
epan2_bofinger.sparsity =  227.9299402797656
epan2_bofinger.bwidth =  .2173486679767846
epan2_bofinger.kbwidth =  91.50878448104551
epan2_bofinger.df_m =  1
epan2_bofinger.df_r =  233
epan2_bofinger.f_r =  .0043873130435281
epan2_bofinger.N =  235
epan2_bofinger.q_v =  582.541259765625
epan2_bofinger.q =  .5
epan2_bofinger.sum_rdev =  46278.05667114258
epan2_bofinger.sum_adev =  17559.93220318131
epan2_bofinger.convcode =  0

epan2_hsheather = Bunch()
epan2_hsheather.table = np.array([
    [.5601805 , .0131763  , 42.51 , 0.000  , .5342206  , .5861403],
    [81.48233 , 14.63518  ,  5.57 , 0.000  , 52.64815  , 110.3165]
    ])
epan2_hsheather.psrsquared     =    0.6206
epan2_hsheather.rank =  2
epan2_hsheather.sparsity =  209.3102085912557
epan2_hsheather.bwidth =  .1574393314202373
epan2_hsheather.kbwidth =  64.53302151153288
epan2_hsheather.df_m =  1
epan2_hsheather.df_r =  233
epan2_hsheather.f_r =  .0047775978378236
epan2_hsheather.N =  235
epan2_hsheather.q_v =  582.541259765625
epan2_hsheather.q =  .5
epan2_hsheather.sum_rdev =  46278.05667114258
epan2_hsheather.sum_adev =  17559.93220318131
epan2_hsheather.convcode =  0

epan2_chamberlain = Bunch()
epan2_chamberlain.table = np.array([
    [.5601805 , .0117925 , 47.50 , 0.000 , .5369469 , .583414],
    [81.48233 , 13.0982 , 6.22 , 0.000 , 55.67629 , 107.2884]
    ])
epan2_chamberlain.psrsquared     =    0.6206
epan2_chamberlain.rank =  2
epan2_chamberlain.sparsity =  187.3286437436797
epan2_chamberlain.bwidth =  .063926976464458
epan2_chamberlain.kbwidth =  25.61257055690209
epan2_chamberlain.df_m =  1
epan2_chamberlain.df_r =  233
epan2_chamberlain.f_r =  .0053382119253919
epan2_chamberlain.N =  235
epan2_chamberlain.q_v =  582.541259765625
epan2_chamberlain.q =  .5
epan2_chamberlain.sum_rdev =  46278.05667114258
epan2_chamberlain.sum_adev =  17559.93220318131
epan2_chamberlain.convcode =  0


rectangle_bofinger = Bunch()
rectangle_bofinger.table = np.array([
    [.5601805 , .0158331 , 35.38 , 0.000 , .5289861 , .5913748],
    [81.48233 , 17.5862 , 4.63 , 0.000 , 46.83404 , 116.1306]
    ])
rectangle_bofinger.psrsquared     =    0.6206
rectangle_bofinger.rank =  2
rectangle_bofinger.sparsity =  251.515372550242
rectangle_bofinger.bwidth =  .2173486679767846
rectangle_bofinger.kbwidth =  91.50878448104551
rectangle_bofinger.df_m =  1
rectangle_bofinger.df_r =  233
rectangle_bofinger.f_r =  .0039759001203803
rectangle_bofinger.N =  235
rectangle_bofinger.q_v =  582.541259765625
rectangle_bofinger.q =  .5
rectangle_bofinger.sum_rdev =  46278.05667114258
rectangle_bofinger.sum_adev =  17559.93220318131
rectangle_bofinger.convcode =  0

rectangle_hsheather = Bunch()
rectangle_hsheather.table = np.array([
    [.5601805 , .0137362 , 40.78 , 0.000 , .5331174 , .5872435],
    [81.48233 , 15.25712 , 5.34 , 0.000 , 51.42279 , 111.5419]
    ])
rectangle_hsheather.psrsquared     =    0.6206
rectangle_hsheather.rank =  2
rectangle_hsheather.sparsity =  218.2051806505069
rectangle_hsheather.bwidth =  .1574393314202373
rectangle_hsheather.kbwidth =  64.53302151153288
rectangle_hsheather.df_m =  1
rectangle_hsheather.df_r =  233
rectangle_hsheather.f_r =  .004582842611797
rectangle_hsheather.N =  235
rectangle_hsheather.q_v =  582.541259765625
rectangle_hsheather.q =  .5
rectangle_hsheather.sum_rdev =  46278.05667114258
rectangle_hsheather.sum_adev =  17559.93220318131
rectangle_hsheather.convcode =  0

rectangle_chamberlain = Bunch()
rectangle_chamberlain.table = np.array([
    [.5601805 , .0118406 , 47.31 , 0.000 , .5368522 , .5835087],
    [81.48233 , 13.1516 , 6.20 , 0.000 , 55.57108 , 107.3936]
    ])
rectangle_chamberlain.psrsquared     =    0.6206
rectangle_chamberlain.rank =  2
rectangle_chamberlain.sparsity =  188.0923150272497
rectangle_chamberlain.bwidth =  .063926976464458
rectangle_chamberlain.kbwidth =  25.61257055690209
rectangle_chamberlain.df_m =  1
rectangle_chamberlain.df_r =  233
rectangle_chamberlain.f_r =  .0053165383171297
rectangle_chamberlain.N =  235
rectangle_chamberlain.q_v =  582.541259765625
rectangle_chamberlain.q =  .5
rectangle_chamberlain.sum_rdev =  46278.05667114258
rectangle_chamberlain.sum_adev =  17559.93220318131
rectangle_chamberlain.convcode =  0

triangle_bofinger = Bunch()
triangle_bofinger.table = np.array([
    [.5601805 , .0138712 , 40.38 , 0.000 , .5328515 , .5875094],
    [81.48233 , 15.40706 , 5.29 , 0.000 , 51.12738 , 111.8373]
    ])
triangle_bofinger.psrsquared     =    0.6206
triangle_bofinger.rank =  2
triangle_bofinger.sparsity =  220.3495620604223
triangle_bofinger.bwidth =  .2173486679767846
triangle_bofinger.kbwidth =  91.50878448104551
triangle_bofinger.df_m =  1
triangle_bofinger.df_r =  233
triangle_bofinger.f_r =  .0045382436463649
triangle_bofinger.N =  235
triangle_bofinger.q_v =  582.541259765625
triangle_bofinger.q =  .5
triangle_bofinger.sum_rdev =  46278.05667114258
triangle_bofinger.sum_adev =  17559.93220318131
triangle_bofinger.convcode =  0

triangle_hsheather = Bunch()
triangle_hsheather.table = np.array([
    [.5601805 , .0128874 , 43.47 , 0.000 , .5347898 , .5855711],
    [81.48233 , 14.31431 , 5.69 , 0.000 , 53.2803 , 109.6844]
    ])
triangle_hsheather.psrsquared     =    0.6206
triangle_hsheather.rank =  2
triangle_hsheather.sparsity =  204.7212998199564
triangle_hsheather.bwidth =  .1574393314202373
triangle_hsheather.kbwidth =  64.53302151153288
triangle_hsheather.df_m =  1
triangle_hsheather.df_r =  233
triangle_hsheather.f_r =  .004884689579831
triangle_hsheather.N =  235
triangle_hsheather.q_v =  582.541259765625
triangle_hsheather.q =  .5
triangle_hsheather.sum_rdev =  46278.05667114258
triangle_hsheather.sum_adev =  17559.93220318131
triangle_hsheather.convcode =  0

triangle_chamberlain = Bunch()
triangle_chamberlain.table = np.array([
    [.5601805 , .0115725 , 48.41 , 0.000 , .5373803 , .5829806],
    [81.48233 , 12.85389 , 6.34 , 0.000 , 56.15764 , 106.807]
    ])
triangle_chamberlain.psrsquared     =    0.6206
triangle_chamberlain.rank =  2
triangle_chamberlain.sparsity =  183.8344452913298
triangle_chamberlain.bwidth =  .063926976464458
triangle_chamberlain.kbwidth =  25.61257055690209
triangle_chamberlain.df_m =  1
triangle_chamberlain.df_r =  233
triangle_chamberlain.f_r =  .0054396769790083
triangle_chamberlain.N =  235
triangle_chamberlain.q_v =  582.541259765625
triangle_chamberlain.q =  .5
triangle_chamberlain.sum_rdev =  46278.05667114258
triangle_chamberlain.sum_adev =  17559.93220318131
triangle_chamberlain.convcode =  0

gaussian_bofinger = Bunch()
gaussian_bofinger.table = np.array([
    [.5601805 , .0197311 , 28.39 , 0.000 , .5213062 , .5990547],
    [81.48233 , 21.91582 , 3.72 , 0.000 , 38.30383 , 124.6608]
    ])
gaussian_bofinger.psrsquared     =    0.6206
gaussian_bofinger.rank =  2
gaussian_bofinger.sparsity =  313.4370075776719
gaussian_bofinger.bwidth =  .2173486679767846
gaussian_bofinger.kbwidth =  91.50878448104551
gaussian_bofinger.df_m =  1
gaussian_bofinger.df_r =  233
gaussian_bofinger.f_r =  .0031904337261521
gaussian_bofinger.N =  235
gaussian_bofinger.q_v =  582.541259765625
gaussian_bofinger.q =  .5
gaussian_bofinger.sum_rdev =  46278.05667114258
gaussian_bofinger.sum_adev =  17559.93220318131
gaussian_bofinger.convcode =  0

gaussian_hsheather = Bunch()
gaussian_hsheather.table = np.array([
    [.5601805 , .016532 , 33.88 , 0.000 , .5276092 , .5927518],
    [81.48233 , 18.36248 , 4.44 , 0.000 , 45.30462 , 117.66]
    ])
gaussian_hsheather.psrsquared     =    0.6206
gaussian_hsheather.rank =  2
gaussian_hsheather.sparsity =  262.6175743002715
gaussian_hsheather.bwidth =  .1574393314202373
gaussian_hsheather.kbwidth =  64.53302151153288
gaussian_hsheather.df_m =  1
gaussian_hsheather.df_r =  233
gaussian_hsheather.f_r =  .0038078182797341
gaussian_hsheather.N =  235
gaussian_hsheather.q_v =  582.541259765625
gaussian_hsheather.q =  .5
gaussian_hsheather.sum_rdev =  46278.05667114258
gaussian_hsheather.sum_adev =  17559.93220318131
gaussian_hsheather.convcode =  0

gaussian_chamberlain = Bunch()
gaussian_chamberlain.table = np.array([
    [.5601805 , .0128123 , 43.72 , 0.000 , .5349378 , .5854232],
    [81.48233 , 14.23088 , 5.73 , 0.000 , 53.44468 , 109.52]
    ])
gaussian_chamberlain.psrsquared     =    0.6206
gaussian_chamberlain.rank =  2
gaussian_chamberlain.sparsity =  203.5280962791137
gaussian_chamberlain.bwidth =  .063926976464458
gaussian_chamberlain.kbwidth =  25.61257055690209
gaussian_chamberlain.df_m =  1
gaussian_chamberlain.df_r =  233
gaussian_chamberlain.f_r =  .004913326554328
gaussian_chamberlain.N =  235
gaussian_chamberlain.q_v =  582.541259765625
gaussian_chamberlain.q =  .5
gaussian_chamberlain.sum_rdev =  46278.05667114258
gaussian_chamberlain.sum_adev =  17559.93220318131
gaussian_chamberlain.convcode =  0

cosine_bofinger = Bunch()
cosine_bofinger.table = np.array([
    [.5601805 , .0121011 , 46.29 , 0.000 , .536339 , .5840219],
    [81.48233 , 13.44092 , 6.06 , 0.000 , 55.00106 , 107.9636]
    ])
cosine_bofinger.psrsquared     =    0.6206
cosine_bofinger.rank =  2
cosine_bofinger.sparsity =  192.2302014415605
cosine_bofinger.bwidth =  .2173486679767846
cosine_bofinger.kbwidth =  91.50878448104551
cosine_bofinger.df_m =  1
cosine_bofinger.df_r =  233
cosine_bofinger.f_r =  .0052020961976883
cosine_bofinger.N =  235
cosine_bofinger.q_v =  582.541259765625
cosine_bofinger.q =  .5
cosine_bofinger.sum_rdev =  46278.05667114258
cosine_bofinger.sum_adev =  17559.93220318131
cosine_bofinger.convcode =  0

cosine_hsheather = Bunch()
cosine_hsheather.table = np.array([
    [.5601805 , .0116679 , 48.01 , 0.000 , .5371924 , .5831685],
    [81.48233 , 12.9598 , 6.29 , 0.000 , 55.94897 , 107.0157]
])
cosine_hsheather.psrsquared     =    0.6206
cosine_hsheather.rank =  2
cosine_hsheather.sparsity =  185.349198428224
cosine_hsheather.bwidth =  .1574393314202373
cosine_hsheather.kbwidth =  64.53302151153288
cosine_hsheather.df_m =  1
cosine_hsheather.df_r =  233
cosine_hsheather.f_r =  .0053952216059205
cosine_hsheather.N =  235
cosine_hsheather.q_v =  582.541259765625
cosine_hsheather.q =  .5
cosine_hsheather.sum_rdev =  46278.05667114258
cosine_hsheather.sum_adev =  17559.93220318131
cosine_hsheather.convcode =  0

cosine_chamberlain = Bunch()
cosine_chamberlain.table = np.array([
    [.5601805 , .0106479 , 52.61 , 0.000 , .539202 , .5811589],
    [81.48233 , 11.82688 , 6.89 , 0.000 , 58.18104 , 104.7836]
])
cosine_chamberlain.psrsquared     =    0.6206
cosine_chamberlain.rank =  2
cosine_chamberlain.sparsity =  169.1463943762948
cosine_chamberlain.bwidth =  .063926976464458
cosine_chamberlain.kbwidth =  25.61257055690209
cosine_chamberlain.df_m =  1
cosine_chamberlain.df_r =  233
cosine_chamberlain.f_r =  .0059120385254878
cosine_chamberlain.N =  235
cosine_chamberlain.q_v =  582.541259765625
cosine_chamberlain.q =  .5
cosine_chamberlain.sum_rdev =  46278.05667114258
cosine_chamberlain.sum_adev =  17559.93220318131
cosine_chamberlain.convcode =  0

parzen_bofinger = Bunch()
parzen_bofinger.table = np.array([
    [.5601805 , .012909 , 43.39 , 0.000 , .5347471 , .5856138],
    [81.48233 , 14.33838 , 5.68 , 0.000 , 53.23289 , 109.7318]
])
parzen_bofinger.psrsquared     =    0.6206
parzen_bofinger.rank =  2
parzen_bofinger.sparsity =  205.0654663067616
parzen_bofinger.bwidth =  .2173486679767846
parzen_bofinger.kbwidth =  91.50878448104551
parzen_bofinger.df_m =  1
parzen_bofinger.df_r =  233
parzen_bofinger.f_r =  .0048764914834762
parzen_bofinger.N =  235
parzen_bofinger.q_v =  582.541259765625
parzen_bofinger.q =  .5
parzen_bofinger.sum_rdev =  46278.05667114258
parzen_bofinger.sum_adev =  17559.93220318131
parzen_bofinger.convcode =  0

parzen_hsheather = Bunch()
parzen_hsheather.table = np.array([
    [.5601805 , .0122688 , 45.66 , 0.000 , .5360085 , .5843524],
    [81.48233 , 13.62723 , 5.98 , 0.000 , 54.63401 , 108.3307]
])
parzen_hsheather.psrsquared     =    0.6206
parzen_hsheather.rank =  2
parzen_hsheather.sparsity =  194.8946558099188
parzen_hsheather.bwidth =  .1574393314202373
parzen_hsheather.kbwidth =  64.53302151153288
parzen_hsheather.df_m =  1
parzen_hsheather.df_r =  233
parzen_hsheather.f_r =  .0051309770185556
parzen_hsheather.N =  235
parzen_hsheather.q_v =  582.541259765625
parzen_hsheather.q =  .5
parzen_hsheather.sum_rdev =  46278.05667114258
parzen_hsheather.sum_adev =  17559.93220318131
parzen_hsheather.convcode =  0

parzen_chamberlain = Bunch()
parzen_chamberlain.table = np.array([
    [.5601805 , .0110507 , 50.69 , 0.000 , .5384084 , .5819526],
    [81.48233 , 12.2743 , 6.64 , 0.000 , 57.29954 , 105.6651]
])
parzen_chamberlain.psrsquared     =    0.6206
parzen_chamberlain.rank =  2
parzen_chamberlain.sparsity =  175.5452813763412
parzen_chamberlain.bwidth =  .063926976464458
parzen_chamberlain.kbwidth =  25.61257055690209
parzen_chamberlain.df_m =  1
parzen_chamberlain.df_r =  233
parzen_chamberlain.f_r =  .0056965359146063
parzen_chamberlain.N =  235
parzen_chamberlain.q_v =  582.541259765625
parzen_chamberlain.q =  .5
parzen_chamberlain.sum_rdev =  46278.05667114258
parzen_chamberlain.sum_adev =  17559.93220318131
parzen_chamberlain.convcode =  0
