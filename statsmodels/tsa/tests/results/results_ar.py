from pathlib import Path

import numpy as np

cur_dir = Path(__file__).resolve().parent


class ARLagResults:
    """
    Results are from R vars::VARselect for sunspot data

    Commands run were

    var_select <- VARselect(SUNACTIVITY, lag.max=16, type=c("const"))
    """

    def __init__(self, type="const"):
        if type == "const":
            ic = [
                6.311751824815273,
                6.321813007357017,
                6.336872456958734,
                551.0094925431335,
                5.647615009344886,
                5.662706783157502,
                5.685295957560077,
                283.61444420963466,
                5.634199640773091,
                5.65432200585658,
                5.684440905060013,
                279.835333966272,
                5.6394157977669,
                5.664568754121261,
                5.702217378125553,
                281.2992674416832,
                5.646102475432464,
                5.676286023057697,
                5.721464371862848,
                283.1872109327845,
                5.628416873122441,
                5.663631012018546,
                5.716339085624555,
                278.2238392848447,
                5.58420418513715,
                5.624448915304128,
                5.684686713710994,
                266.19197555494156,
                5.541163244029505,
                5.586438565467356,
                5.654206088675081,
                254.97935373723556,
                5.483155367013447,
                5.53346127972217,
                5.608758527730753,
                240.61108846854495,
                5.489939895595428,
                5.545276399575022,
                5.628103372384465,
                242.2511993973943,
                5.496713895370946,
                5.557080990621412,
                5.647437688231713,
                243.9003499050695,
                5.503539311586831,
                5.56893699810817,
                5.666823420519329,
                245.57382356198914,
                5.510365149977393,
                5.580793427769605,
                5.686209574981622,
                247.2593969911336,
                5.513740912139918,
                5.589199781203001,
                5.702145653215877,
                248.09965569370948,
                5.515627471325321,
                5.596116931659277,
                5.716592528473011,
                248.5729154848272,
                5.515935627515806,
                5.601455679120634,
                5.729461000735226,
                248.6549279153013,
            ]
            self.ic = np.asarray(ic).reshape(4, -1, order="F")


class ARResultsOLS:
    """
    Results of fitting an AR(9) model to the sunspot data

    Results were taken from Stata using the var command.
    """

    def __init__(self, constant=True):
        self.avobs = 300.0
        if constant:
            self.params = [
                6.7430535917332,
                1.1649421971129,
                -0.40535742259304,
                -0.16653934246587,
                0.14980629416032,
                -0.09462417064796,
                0.00491001240749,
                0.0504665930841,
                -0.08635349190816,
                0.25349103194757,
            ]
            self.bse_stata = [
                2.413485601,
                0.0560359041,
                0.0874490762,
                0.0900894414,
                0.0899348339,
                0.0900100797,
                0.0898385666,
                0.0896997939,
                0.0869773089,
                0.0559505756,
            ]
            self.bse_gretl = [
                2.45474,
                0.0569939,
                0.088944,
                0.0916295,
                0.0914723,
                0.0915488,
                0.0913744,
                0.0912332,
                0.0884642,
                0.0569071,
            ]
            self.rmse = 15.1279294937327
            self.fpe = 236.4827257929261
            self.llf = -1235.559128419549
            filename = Path(cur_dir).joinpath("AROLSConstantPredict.csv")
            predictresults = np.loadtxt(filename)
            fv = predictresults[:300, 0]
            pv = predictresults[300:, 1]
            del predictresults
            self.FVOLSnneg1start0 = fv
            self.FVOLSnneg1start9 = fv
            self.FVOLSnneg1start100 = fv[100 - 9 :]
            self.FVOLSn200start0 = fv[:192]
            self.FVOLSn200start200 = np.hstack((fv[200 - 9 :], pv[: 101 - 9]))
            self.FVOLSn200startneg109 = self.FVOLSn200start200
            self.FVOLSn100start325 = np.hstack((fv[-1], pv))
            self.FVOLSn301start9 = np.hstack((fv, pv[:2]))
            self.FVOLSdefault = fv
            self.FVOLSn4start312 = np.hstack((fv[-1], pv[:8]))
            self.FVOLSn15start312 = np.hstack((fv[-1], pv[:19]))
        elif not constant:
            self.params = [
                1.19582389902985,
                -0.40591818219637,
                -0.15813796884843,
                0.16620079925202,
                -0.08570200254617,
                0.01876298948686,
                0.06130211910707,
                -0.08461507700047,
                0.27995084653313,
            ]
            self.bse_stata = [
                0.055645055,
                0.088579237,
                0.0912031179,
                0.0909032462,
                0.0911161784,
                0.0908611473,
                0.0907743174,
                0.0880993504,
                0.0558560278,
            ]
            self.bse_gretl = [
                0.056499,
                0.0899386,
                0.0926027,
                0.0922983,
                0.0925145,
                0.0922555,
                0.0921674,
                0.0894513,
                0.0567132,
            ]
            self.rmse = 15.29712618677774
            self.sigma = 226.9820074869752
            self.llf = -1239.41217278661
            self.fpe = 241.0221316614273
            filename = Path(cur_dir).joinpath("AROLSNoConstantPredict.csv")
            predictresults = np.loadtxt(filename)
            fv = predictresults[:300, 0]
            pv = predictresults[300:, 1]
            del predictresults
            self.FVOLSnneg1start0 = fv
            self.FVOLSnneg1start9 = fv
            self.FVOLSnneg1start100 = fv[100 - 9 :]
            self.FVOLSn200start0 = fv[:192]
            self.FVOLSn200start200 = np.hstack((fv[200 - 9 :], pv[: 101 - 9]))
            self.FVOLSn200startneg109 = self.FVOLSn200start200
            self.FVOLSn100start325 = np.hstack((fv[-1], pv))
            self.FVOLSn301start9 = np.hstack((fv, pv[:2]))
            self.FVOLSdefault = fv
            self.FVOLSn4start312 = np.hstack((fv[-1], pv[:8]))
            self.FVOLSn15start312 = np.hstack((fv[-1], pv[:19]))


class ARResultsMLE:
    """
    Results of fitting an AR(9) model to the sunspot data using exact MLE

    Results were taken from gretl.
    """

    def __init__(self, constant=True):
        self.avobs = 300
        if constant:
            filename = Path(cur_dir).joinpath("ARMLEConstantPredict.csv")
            filename2 = Path(cur_dir).joinpath("results_ar_forecast_mle_dynamic.csv")
            predictresults = np.loadtxt(filename, delimiter=",")
            pv = predictresults[:, 1]
            dynamicpv = np.genfromtxt(filename2, delimiter=",", skip_header=1)
            self.FVMLEdefault = pv[:309]
            self.FVMLEstart9end308 = pv[9:309]
            self.FVMLEstart100end308 = pv[100:309]
            self.FVMLEstart0end200 = pv[:201]
            self.FVMLEstart200end334 = pv[200:]
            self.FVMLEstart308end334 = pv[308:]
            self.FVMLEstart9end309 = pv[9:310]
            self.FVMLEstart0end301 = pv[:302]
            self.FVMLEstart4end312 = pv[4:313]
            self.FVMLEstart2end7 = pv[2:8]
            self.fcdyn = dynamicpv[:, 0]
            self.fcdyn2 = dynamicpv[:, 1]
            self.fcdyn3 = dynamicpv[:, 2]
            self.fcdyn4 = dynamicpv[:, 3]
        else:
            pass
