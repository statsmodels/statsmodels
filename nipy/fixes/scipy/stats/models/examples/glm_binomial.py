from models.datasets.star98.data import load
from models.functions import add_constant
from models.glm import GLMBinomial
import numpy as np

data = load()
data.exog = add_constant(data.exog)
model = GLMBinomial(data.endog, data.exog)
#results = model.fit()
history = {'params' : [np.inf], 'deviance': [np.inf]}
mu = (model.y + .5)/(model.k + 1)
eta = np.log(mu/(model.k - mu))
iterations = 1
w = mu * (model.k - mu)
