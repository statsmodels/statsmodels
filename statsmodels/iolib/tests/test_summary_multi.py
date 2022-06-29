import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary_multi import pretty_conf_str


def test_pretty_conf_str():
    ser = pd.Series({
        "coef": 0.8951,
        "up": 0.8592,
        "low": 0.9332,
    })

    est_conf = pretty_conf_str(ser, "coef", "up", "low", fmt=".3f")
    assert "0.895 (0.859, 0.933)" == est_conf
    est_conf = pretty_conf_str(ser, "coef", "up", "low", fmt=".4f")
    assert "0.8951 (0.8592, 0.9332)" == est_conf


def test_add_params_summary():
    df_desire = pd.DataFrame(
        {'coef': {'Intercept': -0.07302574743714974,
                  'logpopul': -0.11096689382641269},
         'std err': {'Intercept': 0.08306139388228365,
                     'logpopul': 0.021216468733009933},
         't': {'Intercept': -0.8791779673315302,
               'logpopul': -5.230224464911228},
         'P>|t|': {'Intercept': 0.3793047886781885,
                   'logpopul': 1.693043432353157e-07},
         '[0.025': {'Intercept': -0.23582308795212126,
                    'logpopul': -0.1525504084222323},
         '0.975]': {'Intercept': 0.08977159307782179,
                    'logpopul': -0.06938337923059307},
         'str': {'Intercept': '-0.073 (-0.236, 0.090)',
                 'logpopul': '-0.111 (-0.153, -0.069)'}}
    )

    df = sm.datasets.anes96.load_pandas().data
    res = smf.glm(formula="vote ~ logpopul",
                  data=df, family=sm.families.Binomial()).fit()
    res_no_f = sm.add_params_summary(res)

    assert_frame_equal(df_desire, res_no_f.params_summary)

    df_desire2 = pd.DataFrame(
        {'coef': {'Intercept': -0.07302574743714974,
                  'logpopul': -0.11096689382641269},
         'std err': {'Intercept': 0.08306139388228365,
                     'logpopul': 0.021216468733009933},
         't': {'Intercept': -0.8791779673315302,
               'logpopul': -5.230224464911228},
         'P>|t|': {'Intercept': 0.3793047886781885,
                   'logpopul': 1.693043432353157e-07},
         'f(coef)': {'Intercept': 0.929576895494975,
                     'logpopul': 0.8949683774170433},
         '[0.025': {'Intercept': 0.7899204080091596,
                    'logpopul': 0.8585156164479414},
         '0.975]': {'Intercept': 1.093924395263954,
                    'logpopul': 0.9329689305949442},
         'str': {'Intercept': '0.930 (0.790, 1.094)',
                 'logpopul': '0.895 (0.859, 0.933)'}}
    )

    res_f = sm.add_params_summary(res, func=np.exp)
    assert_frame_equal(df_desire2, res_f.params_summary)


class TestMultiModelSummary:
    @classmethod
    def setup_class(cls):
        df = sm.datasets.anes96.load_pandas().data

        inc = "income"
        df[inc] = (df[inc]
                   .mask(df[inc] <= 24, "17-24")
                   .mask(df[inc] <= 16, "9-16")
                   .mask(df[inc] <= 8, "0-8")
                   )

        res1 = smf.glm(formula="vote ~ logpopul",
                       data=df, family=sm.families.Binomial()).fit()
        res2 = smf.glm(formula="vote ~ logpopul + income",
                       data=df, family=sm.families.Binomial()).fit()

        cls.results = [res1, res2]

    def test_with_func(self):
        df_desire = pd.DataFrame(
            {'Model1': {'logpopul': '0.895 (0.859, 0.933)',
                        'income[T.0-8]': 'Ref.',
                        'income[T.9-16]': np.nan,
                        'income[T.17-24]': np.nan},
             'Model2': {'logpopul': '0.898 (0.861, 0.937)',
                        'income[T.0-8]': 'Ref.',
                        'income[T.9-16]': '1.489 (0.930, 2.386)',
                        'income[T.17-24]': '2.609 (1.685, 4.039)'}}
        )

        results = [sm.add_params_summary(res, func=np.exp, alpha=0.05)
                   for res in self.results]
        df_res = sm.multi_model_summary(
                results,
                accessor=lambda x: x.params_summary["str"],
                columns=None,
                index=["logpopul", 'income[T.0-8]',
                       'income[T.9-16]', 'income[T.17-24]'],
                fill_value="Ref.",
                )
        assert_frame_equal(df_desire, df_res)

    def test_without_func(self):
        df_desire = pd.DataFrame(
            {'M1': {'logpopul': '-0.111 (-0.153, -0.069)',
                    'income[T.0-8]': 'Ref.',
                    'income[T.9-16]': np.nan,
                    'income[T.17-24]': np.nan},
             'M2': {'logpopul': '-0.107 (-0.150, -0.065)',
                    'income[T.0-8]': 'Ref.',
                    'income[T.9-16]': '0.398 (-0.073, 0.870)',
                    'income[T.17-24]': '0.959 (0.522, 1.396)'}}
        )

        results = [sm.add_params_summary(res, func=None, alpha=0.05)
                   for res in self.results]
        df_res = sm.multi_model_summary(
                results,
                accessor=lambda x: x.params_summary["str"],
                columns=["M1", "M2"],
                index=['logpopul', 'income[T.0-8]',
                       'income[T.9-16]', 'income[T.17-24]'],
                fill_value="Ref.",
                )
        assert_frame_equal(df_desire, df_res)


class TestMosaiMultiSummary():

    @classmethod
    def setup_class(cls):
        df = sm.datasets.anes96.load_pandas().data

        inc = "income"
        df[inc] = (df[inc]
                   .mask(df[inc] <= 24, "17-24")
                   .mask(df[inc] <= 16, "9-16")
                   .mask(df[inc] <= 8, "0-8")
                   )

        res1 = smf.glm(formula="vote ~ logpopul",
                       data=df, family=sm.families.Binomial()).fit()
        res2 = smf.glm(formula="vote ~ logpopul + income",
                       data=df, family=sm.families.Binomial()).fit()
        df_60 = df.loc[df["age"].le(60)]
        res3 = smf.glm(formula="vote ~ logpopul",
                       data=df_60, family=sm.families.Binomial()).fit()
        res4 = smf.glm(formula="vote ~ logpopul + income",
                       data=df_60, family=sm.families.Binomial()).fit()

        results = [res1, res2, res3, res4]
        cls.results = [sm.add_params_summary(res, func=np.exp, alpha=0.05)
                       for res in results]

    def test_simple_mosaic(self):
        df_desire = pd.DataFrame(
            {'model1': {('All', 'Intercept'): '0.930 (0.790, 1.094)',
                        ('All', 'logpopul'): '0.895 (0.859, 0.933)',
                        ('All', 'income[T.17-24]'): np.nan,
                        ('All', 'income[T.9-16]'): np.nan,
                        ('Age<=60', 'Intercept'): '0.931 (0.775, 1.118)',
                        ('Age<=60', 'logpopul'): '0.874 (0.834, 0.916)',
                        ('Age<=60', 'income[T.17-24]'): np.nan,
                        ('Age<=60', 'income[T.9-16]'): np.nan},
             'model2': {('All', 'Intercept'): '0.469 (0.310, 0.710)',
                        ('All', 'logpopul'): '0.898 (0.861, 0.937)',
                        ('All', 'income[T.17-24]'): '2.609 (1.685, 4.039)',
                        ('All', 'income[T.9-16]'): '1.489 (0.930, 2.386)',
                        ('Age<=60', 'Intercept'): '0.497 (0.299, 0.826)',
                        ('Age<=60', 'logpopul'): '0.880 (0.839, 0.922)',
                        ('Age<=60', 'income[T.17-24]'): '2.376 (1.402, 4.028)',
                        ('Age<=60', 'income[T.9-16]'): '1.239 (0.690, 2.225)'}}
        )
        df_desire.index.names = ["Row", "index"]
        mosaic = [[0, 1],
                  [2, 3]]
        df_res = sm.mosaic_model_summary(
            self.results, mosaic=mosaic,
            accessor=lambda x: x.params_summary["str"],
            columns=["model1", "model2"],
            rows=["All", "Age<=60"])
        assert_frame_equal(df_desire, df_res)

    def test_none_mosaic(self):
        df_desire = pd.DataFrame(
            {'Case1': {(0, 'Intercept'): -0.07302574743714974,
                       (0, 'logpopul'): -0.11096689382641269,
                       (0, 'income[T.17-24]'): np.nan,
                       (0, 'income[T.9-16]'): np.nan,
                       (1, 'Intercept'): -0.7561815519647701,
                       (1, 'logpopul'): -0.10738092502621271,
                       (1, 'income[T.17-24]'): 0.9588412266442222,
                       (1, 'income[T.9-16]'): 0.3982551394950641,
                       (2, 'Intercept'): -0.07160784355806384,
                       (2, 'logpopul'): -0.13504060656677494,
                       (2, 'income[T.17-24]'): np.nan,
                       (2, 'income[T.9-16]'): np.nan},
             'Case2': {(0, 'Intercept'): np.nan,
                       (0, 'logpopul'): np.nan,
                       (0, 'income[T.17-24]'): np.nan,
                       (0, 'income[T.9-16]'): np.nan,
                       (1, 'Intercept'): np.nan,
                       (1, 'logpopul'): np.nan,
                       (1, 'income[T.17-24]'): np.nan,
                       (1, 'income[T.9-16]'): np.nan,
                       (2, 'Intercept'): -0.6987239090991382,
                       (2, 'logpopul'): -0.12828642341392965,
                       (2, 'income[T.17-24]'): 0.8655012587203116,
                       (2, 'income[T.9-16]'): 0.21407916232401158}})
        df_desire.index.names = ["Row", "index"]
        mosaic = [[0, None],
                  [1, None],
                  [2, 3]]
        df_res = sm.mosaic_model_summary(
            self.results,
            mosaic=mosaic,
            columns=["Case1", "Case2"],
            )

        assert_frame_equal(df_desire, df_res)
