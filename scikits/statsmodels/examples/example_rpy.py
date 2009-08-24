'''Just two examples for using rpy

# example 1: OLS using LM
# example 2: GLM with binomial family

'''

from rpy import r
import numpy as np

from scikits.statsmodels.tools import xi, add_constant
from exampledata import longley, lbw

examples = [1, 2]

if 1 in examples:
    y,x = longley()
    des = np.hstack((np.ones((x.shape[0],1)),x))
    des_cols = ['x.%d' % (i+1) for i in range(x.shape[1])]
    formula = r('y~%s-1' % '+'.join(des_cols))
    frame = r.data_frame(y=y, x=des)
    results = r.lm(formula, data=frame)
    print results.keys()
    print results['coefficients']

    # How to get Standard Errors for COV Matrix?


if 2 in examples:
    #corrected, see also glm_example.py and test_glm.py
    X = lbw()
    X = xi(X, col='race', drop=True)
    des = np.column_stack((X['age'],X['lwt'],X['black'],X['other'],X['smoke'], X['ptl'], X['ht'], X['ui']))
    #des = np.vstack((X['age'],X['lwt'],X['bwt'],X['ftv'],X['smoke'],X['ptl'],X['ht'],X['ui'])).T
    des = np.hstack((des, np.ones((des.shape[0],1))))
    des_cols = ['x.%d' % (i+1) for i in range(des.shape[1])]
    formula = r('y~%s-1' % '+'.join(des_cols))
    frame = r.data_frame(y=X.low, x=des)
    results = r.glm(formula, data=frame, family='binomial')
    params_est = [results['coefficients'][k] for k
                    in sorted(results['coefficients'])]
    print params_est
    print ', '.join(['%13.10f']*9) % tuple(params_est)

# HOW TO DO THIS IN R
# data <- read.csv("./lwb_for_R.csv",headers=FALSE)
# low <- data$V1
# age <- data$V3; lwt <- data$V3; black <- data$V5; other <- data$V6; smoke <- data$V7; ptl <- data$V8; ht <- data$V9; ui <- data$V10
# probably a better way to do that!
# summary(glm(low ~ age + lwt + black + other + smoke + ptl + ht + ui, family=binomial))

