
import numpy as np

class Bunch(dict):
    def __init__(self, **kw):
        dict.__init__(self, kw)
        self.__dict__  = self




#########within
within = Bunch()
within.vcov = np.array([
     0.00012768645836941, -7.03410234815332e-05, -7.03410234815332e-05,
    0.000273587363491536
    ]).reshape(2, 2, order='F')

within.residuals = np.array([
     4.05589834730292, 4.98383927042521, 2.1286670663572, -1.41409888659268,
    -4.48106123301506, -1.65541307385048, 0.0421227714363331,
    7.15322916000645, 6.50698828915109, 1.44445887760994, 2.20388797962284,
    -4.78701683409339, -2.95780106376553, -0.0826736527343734,
    -0.278510705454443, -2.18596060237964, -1.95112942724864,
    -2.81624095662176, -1.59706133272226, -4.31212399343293,
    80.1170268933159, 83.5950142882531, 94.4307161726089, 60.4185275689037,
    36.0019970638347, 34.0020599308991, 40.0214467801983, 7.15742992829584,
    27.9307264146074, 0.53556219279195, -0.600900523653681,
    -17.4204343322977, -30.4645832886125, -24.344379654294,
    -39.8662396295114, -61.4346345167211, -53.1746160402675,
    -65.6852790221196, -77.4318195484994, -93.7876206777322,
    18.8648529270396, 5.14059423307521, -14.0321774799499,
    15.1238377494764, -14.5708012543899, -3.73610551677894,
    1.9654954260025, 7.22066850242186, -10.394106324588, -8.29208870104657,
    6.44799899020349, -22.7951311962427, -3.28688582943348,
    6.50773989006361, -3.91903703329977, 1.49710950161673,
    36.2428485465788, 2.64952351937825, -14.8577756772556,
    -5.77656027287165, -0.0963753732278975, -2.57698129303034,
    -1.71143546795998, 0.677496581183616, -1.62971477642097,
    -2.45007476712497, -1.12413680111979, 0.394976824303755,
    -0.466325144749556, -0.117365229639986, -0.350182235126297,
    0.148243991018648, 2.01381165891012, 3.07127093428706,
    0.64128537047484, -0.326077005390002, -0.719094416379135,
    1.29358982961041, 2.4825399949002, 0.844547325480282, 109.430976746455,
    26.2036246295077, -32.5395114127783, 7.11180639566684,
    -18.3156963201602, 17.2998462527839, 78.0951895780503,
    63.3567285088521, 5.02981521390678, 6.94579959868404, 8.97647380211547,
    44.9996896600479, 58.8192197316567, 36.7060632135786, -15.507155639013,
    -49.009202367479, -37.7249745927728, -61.2814168899284,
    -94.232588120673, -154.364687988501, 47.998480168361,
    -67.5976064785974, -161.021757407447, -44.3624554575648,
    -136.970651532871, -44.1684782738145, 1.95888584987324,
    66.8720353975064, 41.5888249325013, 73.0078738949016, 16.2161523377356,
    93.9718169709394, 14.7382624942549, -44.9130237522317,
    -98.3658135493956, -41.1286053073606, -80.4823532775273,
    -24.3786701657455, 136.283708258883, 250.753374897601, 31.615603324872,
    24.6001374628591, 24.2351681607532, 19.6261581259208, 12.279594790532,
    13.0916666657174, 17.2100877760487, 19.0457082243129,
    -0.391579421012223, 21.044795949318, -1.07801962176703,
    12.2755875017083, 6.14671690951283, -16.2878369100738,
    -25.4525836544759, -16.1194873319667, -14.9514204324739,
    -27.448607752174, -38.6443148898952, -60.7973748777165,
    19.8095462254233, 21.0815179329431, 15.9424672511008, 14.6719090313032,
    1.15159508158265, 2.60496687757132, 17.0083903121398, 10.9913101283255,
    -9.91806567531829, -9.32454468614839, -2.1517728513339,
    -0.135843115392805, -4.39412847028456, 2.62710404546901,
    -8.07205402047406, -24.6742981981222, -11.0241265231021,
    -16.6671298853988, -2.27932885781838, -17.2475146024652,
    44.7110537887614, 28.9632061387617, 32.037333059879, 30.1448878496761,
    14.9918300664756, 15.1680337256156, 19.7045118292542, 4.53831652370106,
    5.73913234512533, 25.6743200167206, -7.50035362494326,
    -10.6167546476463, -13.9236876059855, -18.4004684421604,
    -23.8017829650449, -39.5913201539495, -28.0999581242961,
    -25.1443870197355, -30.9061557834472, -23.6877569767618,
    -58.7244503065472, 38.7242408208849, 36.6417498929441,
    -118.717100521369, -184.00792131641, -61.7186765994492,
    27.6901510800758, 12.2622640196967, -52.4901463088889,
    -99.9982820011162, -113.250785267314, 49.5708468248341,
    38.6294170649394, 118.398075603504, 10.7572777687677, 21.2347108703372,
    128.092202152107, 168.065586121921, 122.053126796745, -83.212286695662,
    48.8287047189399, 26.3718390369558, 10.0181159680939, 13.1085276077745,
    11.8445763421928, 8.68445369037022, 35.6829307831371, 20.2319963444912,
    0.427976450798565, -1.93647758469618, -13.0177853672927,
    0.590420187407162, 14.6327060138582, -6.32239183188838,
    -18.6696305373663, -22.5490967017278, -7.99630255428511,
    -10.9459463352243, -38.0064579892473, -70.9781582422914
    ])
within.df_residual = 207
within.df_resid = 207
within.df_model = 12 # from Stata
within.nobs = 220
# from R, based on normal distribution
#within.conf_int = np.array([
#     0.0879818331472068, 0.277614703609689, 0.132276404904313,
#    0.342452180140319
#    ]).reshape(2, 2, order='F')
# from Stata, based on t distribution
#NOTE: we don't list a constant
within.conf_int = [#(-76.7430990196, -33.7999981334),
                   (.087851586551, .132406651501),
                   (.27742405134, .34264283241)]



within.fittedvalues = np.array([
     -7.96629834730292, -6.18923927042444, 1.25593293364281,
    -1.38830111340732, 0.958661233015066, -0.512986926149512,
    -1.15852277143633, -1.88462916000644, 1.92061171084891,
    0.982141122390064, 0.524712020377161, 1.89461683409339,
    -0.056598936234475, -0.79572634726563, -0.13688929454556,
    0.107560602379636, 1.63472942724863, 3.29684095662175,
    3.76866133272225, 3.74472399343293, -102.239526893316,
    -94.667514288253, -81.9932161726089, -68.7110275689037,
    -55.1544970638347, -49.3245599308991, -40.4239467801983,
    -29.2899299282958, -27.4932264146073, -10.018062192792,
    2.00840052365368, 14.9879343322977, 26.6820832886125, 32.881879654294,
    45.4837396295113, 55.3721345167211, 71.6721160402674, 89.2827790221195,
    107.529319548499, 113.415120677732, -64.6983529270396,
    -18.5040942330752, -5.83132252005002, -49.6473377494764,
    -19.1426987456101, -12.9773944832211, -19.7389954260025,
    -46.5441685024219, -28.329393675412, -18.2614112989534,
    -3.79149899020345, 10.7916311962427, -20.1566141705665,
    -3.27123989006361, -3.22446296670025, 13.0393904983832,
    38.2536514534212, 56.2269764806217, 103.664275677256, 92.1430602728716,
    -0.448124626772102, 1.49248129303034, 0.81693546795998,
    -1.77199658118362, 0.575214776420968, 1.17557476712497,
    0.179636801119793, -1.61947682430376, -1.68817485525044,
    -1.78713477036001, -1.3743177648737, -0.992743991018647,
    -1.28831165891012, -0.49577093428706, 0.484214629525159, 0.66157700539,
    2.30459441637914, 1.62191017038959, 0.9629600050998, 1.19095267451971,
    -178.620976746455, -83.4936246295076, 7.44951141277848,
    -64.8018063956668, -35.8743036798397, -45.1898462527839,
    -67.3851895780503, -73.7467285088521, -46.0198152139068,
    -52.4357995986841, -17.6664738021155, 12.6103103399521,
    -13.9092197316567, 7.30393678642134, 11.5171556390129,
    40.2192023674789, 70.6349745927727, 116.291416889928, 171.442588120673,
    241.674687988501, -338.418480168361, -148.622393521402,
    -36.3982425925531, -305.957544542435, -140.249348467129,
    -102.651521726185, -97.9788858498731, -226.892035397506,
    -150.008824932501, -133.527873894901, -63.0361523377354,
    -13.8918169709393, -53.858262494255, -33.9069762477685,
    45.4458135493955, 76.0086053073604, 228.362353277527, 307.558670165745,
    560.096291741118, 627.926625102399, -46.874603324872, -43.099137462859,
    -35.4741681607532, -40.6251581259207, -25.3885947905319,
    -28.0506666657173, -27.0190877760487, -28.7247082243129,
    -5.80742057898777, -0.463795949317982, 11.509019621767,
    2.78541249829172, 6.28428309048716, 14.9288369100738, 16.1035836544759,
    17.7104873319667, 29.5524204324739, 51.539607752174, 62.8653148898952,
    68.2483748777165, -54.8605462254233, -50.5125179329431,
    -45.4134672511008, -42.5529090313032, -31.9625950815826,
    -29.4759668775713, -29.0093903121398, -23.5923101283255,
    -17.6529343246817, -13.4864553138516, -14.2292271486661,
    -5.10515688460719, 0.833128470284552, 5.99189595453098,
    20.8210540204741, 46.6032981981222, 50.9131265231021, 60.7461298853988,
    74.3883288578184, 97.5565146024652, -67.8765537887614,
    -53.3487061387616, -46.8528330598789, -45.200387849676,
    -35.9373300664755, -29.0535337256155, -23.8000118292541,
    -17.673816523701, -9.05463234512533, -2.46982001672062,
    4.02485362494325, 12.0012546476463, 14.8381876059855, 20.8049684421604,
    26.7962829650448, 34.5258201539495, 45.2744581242961, 50.2288870197355,
    57.1706557834471, 65.6022569767617, -141.850549693453,
    -93.8992408208849, 22.783250107056, -29.4578994786305,
    3.93292131641026, 12.8436765994492, 34.6348489199242, 22.8627359803033,
    3.61514630888885, -22.2767179988838, -38.5242147326865,
    -39.745846824834, -28.6044170649395, -34.3730756035042,
    -16.1322777687677, -12.9097108703373, 49.6327978478929,
    66.9594138780794, 108.471873203255, 132.037286695662,
    -78.7902047189399, -43.3633390369558, -17.8596159680939,
    -33.1100276077745, -35.8960763421928, -23.0059536903702,
    -30.0644307831371, -19.7834963444912, -6.29947645079857,
    -3.14502241530382, 9.39628536729267, 9.97807981259285,
    -1.96420601385819, 12.9908918318884, 7.81813053736626,
    11.8975967017278, 19.4848025542851, 39.8344463352243, 85.1949579892473,
    96.6866582422914
    ])
within.params = np.array([
     0.11012911902576, 0.310033441875004
    ])
within.bse = np.array([
     0.0112998432895952, 0.016540476519482
    ])
within.tvalues = np.array([
     9.7460748970887, 18.7439244274392
    ])
within.pvalues = np.array([
     1.03389477552542e-18, 1.74637965655695e-46
    ])
within.rsquared = 0.766670651548843
within.rsquared_adj = 0.721367385775502
within.fvalue = 340.079004043144
within.f_pvalue = 3.8444093182938e-66
# from stata
within.ess = 1720828.222530854
within.ssr = 523718.6621769458


#########swar1w
swar1w = Bunch()
swar1w.vcov = np.array([
     660.334575959393, -0.0817625586545722, -0.00983783012338412,
    -0.0817625586545722, 9.82836972741121e-05, -5.989113766829e-05,
    -0.00983783012338412, -5.989113766829e-05, 0.000268543702465508
    ]).reshape(3, 3, order='F')

swar1w.residuals = np.array([
     8.74470957815893, 9.6856565281138, 6.88566330507811, 3.32143710026088,
    0.27140763496074, 3.08585945571753, 4.77820676010704, 11.8833990064895,
    11.2646278357799, 6.1945516451601, 6.95070251520215,
    -0.0303168933642627, 1.78405150078327, 4.65350294506109,
    4.46184810262007, 2.55557685663715, 2.80134273829872, 1.9482576999242,
    3.17045706480129, 0.454065759217415, 71.0381259166071,
    74.5660623638419, 85.4862766598297, 51.5555224078768, 27.2302967654333,
    25.2672451791515, 31.3455864230022, -1.44785292495172,
    19.3326417289438, -7.94576579251518, -9.00166125336096,
    -25.7354933048738, -38.7024617067206, -32.5451154397769,
    -47.9843240093161, -69.4886920950311, -61.1136937120691,
    -73.5072299002354, -85.1317583504575, -101.453376951189,
    22.2265000172516, 8.84788699358315, -10.2379706585868,
    18.5847837195284, -10.8856917365108, -0.00577825366469464,
    5.64263744591678, 10.6985211104136, -6.77861736356787,
    -4.59916422531629, 10.2510617304168, -18.8926981793938,
    0.380182623116249, 10.2968054777319, -0.141590351492696,
    5.39113194480648, 40.3125162308674, 6.82561418189994,
    -10.3446941887584, -1.37179735942506, 6.60882789847086,
    4.14267082110953, 5.00320837199215, 7.37277752493154, 5.08318203958775,
    4.267367646951, 5.58588460682507, 7.09163354084473, 6.22989812216267,
    6.57817247882373, 6.34847566258041, 6.84978193286564, 8.71292308429443,
    9.77583466178357, 7.35253302936162, 6.38606303375112, 6.00522979041108,
    8.01253062203805, 9.19598856151047, 7.55884873958921, 82.8522266679566,
    0.334336784938206, -57.7328886133988, -18.6343263880099,
    -43.8507178798355, -8.30936320276239, 52.3089154614214,
    37.5013433050274, -20.6284909471339, -18.7609507998941,
    -16.4696429650944, 19.7715597889579, 33.3571941872193,
    11.3747278898078, -40.831073116255, -74.1277812372509,
    -62.6237225313945, -85.8562702129655, -118.418764748659,
    -178.054010410641, 44.0502121532865, -70.1421564132953,
    -162.760389319801, -48.1343119764532, -139.50108691244,
    -46.4188914322412, -0.272017994783243, 63.6612117835623,
    38.9658555115045, 70.5282966228149, 14.2434792381246, 92.3226175766586,
    12.6744981562208, -46.8793138547624, -99.7699513629828,
    -42.3295082589521, -80.57857050718, -23.9541555399982, 138.48571204041,
    253.318310910048, 26.7276959255942, 19.7366109558838, 19.4257833997526,
    14.7734160310502, 7.537610074337, 8.32494259002351, 12.447541104523,
    14.2658985254577, -5.00832512443226, 16.4654484812762,
    -5.57874549658304, 7.71411039878187, 1.60143856622233,
    -20.7768162791013, -29.9363146739069, -20.5931286058945,
    -19.3413060559594, -31.681421627778, -42.8017390572887,
    -64.9225754386073, 23.861269160214, 25.1627734306872, 20.0580366909236,
    18.8051764066904, 5.3610573170184, 6.83177516326508, 21.2357930723871,
    15.2531214317904, -5.61629798972089, -4.99103114994518,
    2.17628101547725, 4.25985066515326, 0.0404517858689355,
    7.09512580275832, -3.50015115121609, -19.9143934694446,
    -6.23610107858415, -11.8128858746626, 2.6732635172844,
    -12.130367517499, 42.5972660973069, 26.950112599729, 30.0672325984147,
    28.1798447038423, 13.0865220735109, 13.3051779687534, 17.8715593711857,
    2.74340397703572, 4.00097248356941, 23.9796576581533,
    -9.15222492353024, -12.2137858223585, -15.5067045613844,
    -19.9439353266056, -25.3059988714623, -41.0450521539049,
    -29.4800753302785, -26.4926877711011, -32.2087393125556,
    -24.9353449233678, -37.3605717730726, 60.4478735989891,
    59.2164585721188, -96.5788934209778, -161.636830714924,
    -39.2621085197047, 50.3074081755553, 34.7794606215936,
    -30.1179257840961, -77.8124380910279, -91.1654685207317,
    71.6731508125139, 60.772528496533, 140.484490108805, 32.965918786732,
    43.4653021379321, 150.795684526791, 190.865827731754, 145.106175495378,
    -59.9977928146921, 47.8592377738015, 25.6676991309843,
    9.50262929616695, 12.4755199014307, 11.1889906123293, 8.12432534154263,
    35.0668816664064, 19.6849373383352, -0.0258105074288305,
    -2.36885597546602, -13.356736497625, 0.257880232921106,
    14.2027578693679, -6.64674460946478, -19.0362808852023,
    -22.8835903357735, -8.27178904932139, -11.074293933959,
    -37.804920208062, -70.7031099668722
    ])
swar1w.df_residual = 217
swar1w.df_resid = 217
swar1w.df_model = 2
swar1w.nobs = 220
swar1w.conf_int = np.array([
     -104.30874886543, 0.0898745975235046, 0.275917502169502,
    -3.57845389061083, 0.128736032176593, 0.340154549878063
    ]).reshape(3, 2, order='F')

swar1w.fittedvalues = np.array([
     -11.6868545697147, -9.92280151966954, -2.53280829663414,
    -5.15558209181672, -2.82555262651658, -4.28600444727337,
    -4.92635175166287, -5.64654399804533, -1.86877282733573,
    -2.79969663671594, -3.25384750675799, -1.89382809819158,
    -3.8301964923391, -4.56364793661693, -3.9089930941759,
    -3.66572184819298, -2.14948772985456, -0.499402691480038,
    -0.0306020563571225, -0.0532107507732533, -84.4227338309565,
    -76.9006702781913, -64.310884574179, -51.1101303222261,
    -37.6449046797826, -31.8518530935009, -23.0101943373515,
    -11.9467549893976, -10.1572496432931, 7.20115787816584,
    19.1470533390116, 32.0408853905244, 43.6578537923712, 49.8205075254276,
    62.3397160949668, 72.1640841806818, 88.3490857977198, 105.842621985886,
    123.967150436108, 129.818769036839, -55.8835047454012,
    -10.0348917217328, 2.55096593043715, -40.931788447678,
    -10.6513129916389, -4.53122647448493, -11.2396421740664,
    -37.8455258385632, -19.7683873645817, -9.77784050283333,
    4.5819335414336, 19.0656934512441, -11.6471873512659, 5.11618979411848,
    5.17458562334309, 21.3218633270439, 46.360479040983, 64.2273810899505,
    111.327689460609, 99.9147926312755, -6.71722857986417,
    -4.79107150250285, -5.46160905338546, -8.03117820632484,
    -5.70158272098106, -5.10576832834431, -6.09428528821838,
    -7.88003422223804, -7.94829880355598, -8.04657316021704,
    -7.63687634397372, -7.25818261425894, -7.55132376568773,
    -6.76423534317689, -5.79093371075493, -5.61446371514443,
    -3.98363047180439, -4.66093130343136, -5.31438924290378,
    -5.08724942098252, -137.58004501768, -43.1621551346619,
    47.1050702636751, -24.5934919617138, 4.12289953011184,
    -5.11845514696127, -27.1367338111451, -33.4291616547511,
    -5.89932740258975, -12.2668675498295, 22.2418246153708,
    52.3006218613184, 26.014987463057, 47.0974537604686, 51.3032547665314,
    79.7999628875273, 109.995904181671, 155.328451863242, 210.090946398936,
    279.826192060917, -248.505839418894, -60.1134708523119,
    51.3047620541941, -216.221315289154, -51.7545403531676,
    -14.436735833366, -9.78360927082399, -137.716839049169,
    -61.4214827771116, -45.0839238884221, 24.9008934962682,
    73.7217551577341, 34.169874578172, 54.0236865891553, 132.814324097376,
    163.173880993345, 314.422943241573, 393.098528274391, 643.858660693983,
    711.326061824345, -36.0642565165764, -32.313171546866,
    -24.7423439907348, -29.8499766220325, -14.7241706653192,
    -17.3615031810058, -16.3341016955052, -18.0224591164399,
    4.73176453345002, 10.0379909277415, 21.9321849056008, 13.2693290102359,
    16.7520008427954, 25.340255688119, 26.5097540829247, 28.1065680149122,
    39.8647454649772, 61.6948610367958, 72.9451784663065, 78.296014847625,
    -51.0780336785104, -46.7595379489836, -41.6948012092199,
    -38.8519409249867, -28.3378218353148, -25.8685396815615,
    -25.4025575906834, -20.0198859500868, -14.1204665285755,
    -9.98573336835119, -10.7230455337736, -1.66661518344964,
    4.23278369583469, 9.3581096789453, 24.0833866329197, 49.6776289511482,
    53.9593365602878, 63.7261213563662, 77.2699719644192, 100.273602999203,
    -59.0335182066458, -44.6063647090679, -38.1534847077536,
    -36.5060968131811, -27.3027741828497, -20.4614300780923,
    -15.2378114805245, -9.14965608637457, -0.587224592908267,
    5.95409023250789, 12.4059728141914, 20.3275337130196, 23.1504524520455,
    29.0776832172667, 35.0297467621235, 42.708800044566, 53.3838232209397,
    58.3064356617623, 65.2024872032167, 73.579092814029, -105.179781507823,
    -57.5882268798848, 58.2431881469855, 6.43854014008219,
    39.5964774340284, 48.4217552388091, 70.052238543549, 58.3801860975107,
    39.2775725032005, 13.5720848101323, -2.57488476016396,
    -3.81350409340948, 7.28711822257134, 1.57515661029921,
    19.6937279323724, 22.8943445811723, 84.9639621923138, 102.193818987351,
    143.453471223726, 166.857439533797, -71.756560784332,
    -36.5950221415149, -11.2799523066975, -26.4128429119613,
    -29.1763136228599, -16.3816483520732, -23.384204676937,
    -13.1722603488657, 0.218487496898278, 3.35153296493546,
    15.7994134870944, 16.3747967565483, 4.52991912010151, 19.3794215989342,
    14.2489578746718, 18.296267325243, 25.8244660387908, 46.0269709234285,
    91.0575971975314, 102.475786956342
    ])
swar1w.params = np.array([
     -53.9436013780205, 0.109305314850049, 0.308036026023783
    ])
swar1w.bse = np.array([
     25.6969760080713, 0.00991381345770194, 0.0163873030870094
    ])
swar1w.tvalues = np.array([
     -2.09921982108233, 11.025556948031, 18.7972373726321
    ])
swar1w.pvalues = np.array([
     0.0369535275437917, 9.82125083138092e-23, 2.00286481277529e-47
    ])
swar1w.rsquared = 0.769987793295529
swar1w.rsquared_adj = 0.75948795975059
swar1w.fvalue = 363.214095328015
swar1w.f_pvalue = 5.62303742810564e-70
swar1w.ssr = 550607.132648455
swar1w.centered_tss = 2393817.00883334
swar1w.ess = swar1w.centered_tss - swar1w.ssr

#########pooling
pooling = Bunch()
pooling.vcov = np.array([
     70.78481025337, -0.0124918399266559, -0.0831847528010257,
    -0.0124918399266559, 3.04575112267226e-05, -6.85227320394832e-05,
    -0.0831847528010257, -6.85227320394832e-05, 0.000587008133873853
    ]).reshape(3, 3, order='F')

pooling.residuals = np.array([
     26.0462581530056, 26.9877848569895, 23.9862941286514, 21.0450711804988,
    18.0680903906862, 21.0080393270655, 22.8375186110635, 30.1200510930617,
    29.6150932017628, 24.7454887645672, 25.4808806485672, 18.5419646434891,
    20.5234585741178, 23.4715304541005, 23.4694609591541, 21.7439730621012,
    22.0619961187195, 21.2503789210599, 22.6010400597158, 20.2260872034233,
    18.3473971388799, 23.4968528247342, 36.8630417318727, 7.60664887972952,
    -14.3716370590052, -14.6209295412348, -6.70794883521659,
    -36.2679597280292, -13.7250676603897, -37.6996849591419,
    -36.5455252264851, -50.524050677744, -60.9985059398109,
    -52.4020694493319, -65.0013692282183, -84.032251924139,
    -74.3739534905429, -83.2796478431481, -91.4846236503067,
    -105.171282221685, 28.493059111182, 12.8925205754808,
    -4.46160963528091, 28.0702247205556, -1.65811082478727,
    9.19574676286199, 15.9366757114027, 21.903375661065, 3.15183696654327,
    4.22465026400356, 17.8258978791751, -9.13395751913467, 12.751290450012,
    23.1424288036153, 16.2448373951684, 22.5101679487838, 60.0726297613967,
    34.0279671928742, 19.8912505784395, 35.9638792266782, 31.8046087403342,
    29.2663105718977, 30.1455897931557, 32.6371317779007, 30.2189445196078,
    29.3584331482983, 30.7156664704612, 32.2792188807956, 31.3969346631422,
    31.7337164969501, 31.4748944265723, 31.9504637571071, 33.8914513319472,
    35.0576913062112, 32.7792965777015, 31.9329827632025, 31.5055673632523,
    33.627191424431, 35.0086116051647, 33.6130393890848, -84.8147528326243,
    -171.220789677835, -232.310792656181, -186.143392657272,
    -211.17131390809, -173.854250652471, -108.915291545364,
    -117.049079807735, -173.438129427801, -171.13261182431,
    -170.654101155781, -133.336067290229, -107.976472114274,
    -122.680053851657, -167.952230799708, -199.840182523208,
    -187.50399855677, -207.685067328569, -235.799196007171,
    -290.330640687916, 2.77897790663982, -115.682029064167,
    -203.694979286927, -71.2887494769114, -171.075933687885,
    -79.4170012125881, -28.9203437878969, 45.7530870141238,
    13.6356260924958, 38.4628703431028, -15.130587182464, 73.6824130114561,
    30.1526162231919, -15.0239667114942, -62.3771554989506,
    1.12677628437592, -34.0033318705438, 40.0808129962684,
    223.560065043181, 377.935943338475, -5.10092024357504,
    -11.1283569316788, -10.9440425977942, -13.9231961730965,
    -20.9300242323662, -18.5876544117087, -13.4663675517277,
    -10.2336287801426, -28.1196816145727, -6.13768805724793,
    -25.5121150450817, -13.1384496395606, -16.4647303845168,
    -36.7771297727698, -44.9447375847027, -35.1063933030775,
    -32.9881990909195, -44.1817095486999, -53.0777476854715,
    -73.0731120194898, 34.7279426572245, 36.7087542615694,
    32.4952963209899, 32.2019038004345, 19.150225918425, 20.8743222178573,
    36.1133701474326, 31.6830995229307, 11.8555446052303, 12.1115458580642,
    19.2855528374918, 21.1168628656236, 18.2536809096073, 26.5872930473903,
    17.3737251980723, 1.17347794943333, 15.866240617062, 11.9941060429582,
    26.9277993836624, 13.6147173978842, 24.2373965108286, 10.2624622542319,
    14.7283728395578, 14.9245325407954, 2.22009583045889, 4.78821932848389,
    11.8862423862943, -1.22791991437981, 1.90622338820959, 23.277097528567,
    -8.44879194431889, -10.4604415904878, -11.7546262625086,
    -14.9703327673996, -18.969173008573, -32.9037012491738,
    -19.839461965282, -15.5348612850353, -19.7423328663403,
    -10.4786901594048, 80.0281778661319, 175.245543249621,
    174.912320033613, 35.1314098094683, -26.5117217937513,
    89.8682149955064, 179.08881042086, 167.672365059802, 103.984126883047,
    55.3569804804481, 36.556055501569, 191.718978541407, 192.880423519594,
    276.876001472461, 172.701061367118, 183.685559350624, 286.551047522953,
    335.522775931993, 304.878593710007, 103.046399156664, 28.9971980438674,
    5.02831137246906, -11.7191011774235, -7.00300871721549,
    -7.64294329325444, -11.033917492858, 17.1676362684794,
    3.64051063139649, -14.4627470601755, -16.3079195625156,
    -27.7769836258497, -14.7995438804781, 2.10295598342333,
    -17.5993994323588, -28.6766101421832, -33.2033543606251,
    -19.6185000445174, -21.8823943585963, -47.9761774129078,
    -77.7341160018389
    ])
pooling.df_residual = 217
pooling.df_resid = 217
pooling.nobs = 220
pooling.conf_int = np.array([
     -54.8999579800172, 0.103717650240182, 0.180027626692945,
    -21.9201499927673, 0.12535107578107, 0.275000624406798
    ]).reshape(3, 2, order='F')

pooling.fittedvalues = np.array([
     -23.108258153004, -21.3447848569938, -13.7532941286525,
    -16.9990711804989, -14.7420903906863, -16.3280393270657,
    -17.1055186110636, -18.0030510930618, -14.3390932017629,
    -15.4704887645673, -15.9038806485673, -14.5859646434892,
    -16.6894585741179, -17.5015304541006, -17.0364609591542,
    -16.9739730621013, -15.5299961187196, -13.92137892106,
    -13.5810400597159, -13.9450872034234, 21.33260286112, 27.2331471752657,
    37.3769582681273, 45.9033511202705, 57.0216370590052, 61.1009295412348,
    68.1079488352166, 75.9379597280292, 75.9650676603897, 90.019684959142,
    99.7555252264851, 109.894050677744, 119.018505939811, 122.742069449332,
    132.421369228218, 139.772251924139, 154.673953490543, 168.679647843148,
    183.384623650307, 186.601282221685, 11.7969408888179, 59.8674794245192,
    70.7216096352809, 23.5297752794444, 54.0681108247873, 60.214253237138,
    52.4133242885972, 24.896624338935, 44.2481630334567, 55.3453497359964,
    70.9541021208249, 83.2539575191347, 49.928709549988, 66.2175711963847,
    62.7351626048316, 78.1498320512162, 100.547370238603, 110.972032807126,
    155.038749421561, 136.526120773322, -29.2646087403343,
    -27.2663105718978, -27.9555897931558, -30.6471317779008,
    -28.1889445196079, -27.5484331482984, -28.5756664704613,
    -30.4192188807957, -30.4669346631423, -30.5537164969502,
    -30.1148944265724, -29.7104637571072, -30.0814513319473,
    -29.3976913062113, -28.5692965777016, -28.5129827632026,
    -26.8355673632525, -27.6271914244312, -28.4786116051648,
    -28.493039389085, 117.914752832624, 216.220789677835, 309.510792656181,
    230.743392657272, 259.271313908091, 248.254250652471, 221.915291545364,
    208.949079807735, 234.738129427801, 227.93261182431, 264.254101155781,
    293.236067290229, 255.176472114274, 268.980053851657, 266.252230799708,
    293.340182523208, 322.70399855677, 364.985067328569, 415.299196007171,
    479.930640687916, 314.821022093361, 507.482029064168, 614.294979286928,
    328.988749476912, 501.875933687885, 540.617001212589, 540.920343787898,
    402.246912985877, 485.964373907505, 509.037129656898, 576.330587182465,
    614.417586988545, 538.747383776809, 544.223966711495, 617.477155498951,
    641.773223715625, 789.903331870545, 851.119187003733, 1080.83993495682,
    1108.76405666153, 31.730920243575, 34.5183569316788, 41.5940425977941,
    34.8131961730964, 49.7100242323662, 45.5176544117087, 45.5463675517277,
    42.4436287801425, 63.8096816145727, 68.6076880572479, 77.8321150450817,
    70.0884496395606, 70.7847303845168, 77.3071297727698, 77.4847375847027,
    78.5863933030775, 89.4781990909195, 110.1617095487, 119.187747685472,
    122.41311201949, -14.3679426572247, -10.7287542615695,
    -6.55529632099004, -4.67190380043452, 5.44977408157493,
    7.66567778214269, 7.29662985256732, 11.1269004770692, 15.9844553947697,
    20.4884541419358, 19.7444471625081, 29.0531371343764, 33.5963190903926,
    37.4427069526097, 50.7862748019276, 76.1665220505667, 79.433759382938,
    87.4958939570418, 100.592200616338, 122.105282602116,
    0.192603489171354, 12.9475377457681, 18.0516271604421,
    17.6154674592045, 24.429904169541, 28.921780671516, 31.6137576137057,
    35.6879199143797, 42.3737766117904, 47.5229024714329, 52.5687919443188,
    59.4404415904878, 60.2646262625086, 64.9703327673996, 69.559173008573,
    75.4337012491738, 84.609461965282, 88.2148612850354, 93.6023328663403,
    99.9886901594048, 129.871822133868, 180.054456750379, 294.987679966387,
    227.168590190532, 256.911721793752, 271.731785004494, 293.71118957914,
    277.927634940199, 257.615873116953, 232.843019519552, 222.143944498431,
    228.581021458593, 227.619576480406, 217.62399852754, 232.398938632882,
    235.114440649376, 301.648952477048, 309.977224068007, 336.121406289993,
    356.253600843337, -16.0671980438675, 20.8716886275309,
    46.7691011774234, 29.8930087172155, 26.4829432932544, 39.603917492858,
    31.3423637315205, 39.6994893686035, 51.4827470601755, 54.1179195625156,
    67.0469836258497, 68.2595438804781, 53.4570440165767, 67.1593994323588,
    60.7166101421832, 65.4433543606251, 73.9985000445175, 93.6623943585963,
    138.056177412908, 146.334116001839
    ])
pooling.params = np.array([
     -38.4100539863922, 0.114534363010626, 0.227514125549872
    ])
pooling.bse = np.array([
     8.41337092094304, 0.00551883241516923, 0.0242282507390413
    ])
pooling.tvalues = np.array([
     -4.56535844518393, 20.7533685378476, 9.39044787014924
    ])
pooling.pvalues = np.array([
     8.35043582578056e-06, 1.96092517789725e-53, 8.50196596399461e-18
    ])
pooling.rsquared = 0.817887031542023
pooling.rsquared_adj = 0.80673402656645
pooling.fvalue = 487.284039537178
pooling.f_pvalue = 5.58450612030346e-81
pooling.ssr = 1768678.40150083
pooling.centered_tss = 9711984.9095698
pooling.ess = pooling.centered_tss - pooling.ssr

#########between
between = Bunch()
between.vcov = np.array([
     1635.68983701992, 0.125738762522479, -4.60770363269256,
    0.125738762522479, 0.000722778784396992, -0.0032681210176942,
    -4.60770363269256, -0.0032681210176942, 0.0304871067512499
    ]).reshape(3, 3, order='F')

between.residuals = np.array([
     4.46596667876909, 23.5783271554474, -3.39874339861887,
    0.744713027305126, -163.507400160883, 12.8215935087958,
    -4.48144887217282, 3.18416226901898, 25.466346479501, 143.698633049183,
    -42.5721497363463
    ])
between.df_residual = 8
between.df_model = 2 # from Stata
between.df_resid = 8
between.nobs = 11
#R using normal distribution for between. Stata uses t. We prefer t here.
#between.conf_int = np.array([
#     -86.6506046370478, 0.0819060157392942, -0.312532633876553,
#    71.8856391981069, 0.187291497409824, 0.371908642339404
#    ]).reshape(3, 2, order='F')
# from Stata
between.conf_int = [
                    (-100.6457,    85.88077),
                    ( .0726029,    .1965946),
                    (-.3729532,    .4323292)]



between.fittedvalues = np.array([
     2.38243332123098, 38.2241728445525, 89.5222433986189, 2.33978697269486,
    265.797400160883, 595.198406491204, 46.3704488721728, 52.226837730981,
    22.129153520499, 266.776366950817, 85.4636497363463
    ])
between.params = np.array([
     -7.38248271947046, 0.134598756574559, 0.0296880042314253
    ])
between.bse = np.array([
     40.4436625074921, 0.0268845454563954, 0.174605574800033
    ])
between.tvalues = np.array([
     -0.182537442500488, 5.0065476015902, 0.170028959644762
    ])
between.pvalues = np.array([
     0.859701860053786, 0.00104425263998374, 0.869208517720077
    ])
between.rsquared = 0.864404649700028
between.rsquared_adj = 0.628657927054566
between.fvalue = 25.4995366076414
between.f_pvalue = 0.00033804863728254
between.ssr = 50627.49374122482
between.ess = 322744.4075018751
between.centered_tss = 373371.9012431
