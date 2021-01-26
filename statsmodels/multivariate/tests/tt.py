import numpy as np

corr = 0.75*np.eye(4) + 0.25*np.ones((4, 4))
n_factor = 1
k_endog = 4

u, s, _ = np.linalg.svd(corr, 0)
u *= np.sqrt(s)
u = u[:, 0:n_factor]
f = 1 - s[0:n_factor].sum() / k_endog
start1 = f * np.ones(k_endog)
start = np.concatenate((start1, u.T.flat))

uniq = start[0:k_endog]**2
load = start[k_endog:]

c0 = np.diag(uniq) + np.outer(load, load)
