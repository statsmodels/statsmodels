import os

import pandas as pd

from statsmodels.genmod.generalized_linear_model import GLM

model_families = [("abc",families.Gaussian()), ("cde",families.Binomial())]

family = model_families[0]

family[0] + "_asdf"

GLM()

test_data = pd.read_csv(f"{os.getcwd()}/statsmodels/genmod/tests/test_data/test_data.csv")