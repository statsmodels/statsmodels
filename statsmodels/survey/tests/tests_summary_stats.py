from summary_stats import SurveyStat
import unittest
import pandas as pd
import numpy as np

df = pd.read_csv("examples/survey_df.csv")
data = df.copy()
n, p = data.shape
svy = SurveyStat(df, cluster="dnum", prob_weights="pw")

class TestSummary(unittest.TestCase):
    def test_intialization(self):
        np.testing.assert_equal(data["dnum"].reshape(n,1), svy.cluster)
        np.testing.assert_equal(data["pw"].reshape(n, 1), svy.prob_weights)
        self.assertEqual(None, svy.strata)


suite = unittest.TestLoader().loadTestsFromTestCase(TestSummary)
unittest.TextTestRunner(verbosity=2).run(suite)