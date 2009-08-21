from rpy import r
import numpy as np

# OLS Example using LM

#FIXME: longley(), lbw() are not defined, copied exampledata.py and datafiles
#FIXME: still errors in loading lbw data: field named black not found.

from exampledata import longley, lbw

example = 1

if example == 1:
    y,x = longley()
    des = np.hstack((np.ones((x.shape[0],1)),x))
    des_cols = ['x.%d' % (i+1) for i in range(x.shape[1])]
    formula = r('y~%s-1' % '+'.join(des_cols))
    frame = r.data_frame(y=y, x=des)
    results = r.lm(formula, data=frame)
    print results.keys()
    print results['coefficients']

    # How to get Standard Errors for COV Matrix?


if example == 2:
    X = lbw()
    des = np.vstack((X['age'],X['lwt'],X['black'],X['other'],X['smoke'],X['ptl'],X['ht'],X['ui'])).T
    des = np.hstack((np.ones((des.shape[0],1)),des))
    des_cols = ['x.%d' % (i+1) for i in range(des.shape[1])]
    formula = r('y~%s-1' % '+'.join(des_cols))
    frame = r.data_frame(y=X.low, x=des)
    results = r.glm(formula, data=frame, family='binomial')

# HOW TO DO THIS IN R
# data <- read.csv("./lwb_for_R.csv",headers=FALSE)
# low <- data$V1
# age <- data$V3; lwt <- data$V3; black <- data$V5; other <- data$V6; smoke <- data$V7; ptl <- data$V8; ht <- data$V9; ui <- data$V10
# probably a better way to do that!
# summary(glm(low ~ age + lwt + black + other + smoke + ptl + ht + ui, family=binomial))

