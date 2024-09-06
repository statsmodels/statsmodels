import pandas as pd
import numpy as np
from datasplitter import train_test_split

'''
Main function:

train_test_split(X,Y)

for train test 4 outputs
for train test val 6 outputs

Arguments
val_split = True/False
shuffle = True/False
stratify = True/False

train_size = 0.7 or 70 (percent or number)
test_size = 0.2 or 20 (percent or number)
val_size = 0.1 or 10 (percent or number)


'''

# List input
X1=[1,2,3,4,5,6,7,8,5,6,7,8,9,10,1,2,3,4,5,6,7,8,5,6,7,8,9,10]
Y1=[1,2,1,2,1,2,1,2,1,2,1,1,1,1,1,2,1,2,1,2,1,2,1,2,1,1,1,1]

# Pandas input
X2=pd.DataFrame(list(zip(X1, Y1)))
Y2=X2[1]

# Array input
X3=np.transpose(np.array([X1,X1]))
Y3=np.transpose(np.array(Y1))




xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X1,Y1,val_split=True,shuffle=True,stratify=True)
xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X1,Y2,val_split=True,shuffle=True,stratify=True)
xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X1,Y3,val_split=True,shuffle=True,stratify=True)

xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X2,Y1,val_split=True,shuffle=True,stratify=True)
xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X2,Y2,val_split=True,shuffle=True,stratify=True)
xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X2,Y3,val_split=True,shuffle=True,stratify=True)

xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X3,Y1,val_split=True,shuffle=True,stratify=True)
xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X3,Y2,val_split=True,shuffle=True,stratify=True)
xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X3,Y3,val_split=True,shuffle=True,stratify=True)


xtrain,xtest,ytrain,ytest=train_test_split(X1,Y1,shuffle=True,stratify=True)
xtrain,xtest,ytrain,ytest=train_test_split(X1,Y2,shuffle=True,stratify=True)
xtrain,xtest,ytrain,ytest=train_test_split(X1,Y3,shuffle=True,stratify=True)

xtrain,xtest,ytrain,ytest=train_test_split(X2,Y1,shuffle=True,stratify=True)
xtrain,xtest,ytrain,ytest=train_test_split(X2,Y2,shuffle=True,stratify=True)
xtrain,xtest,ytrain,ytest=train_test_split(X2,Y3,shuffle=True,stratify=True)

xtrain,xtest,ytrain,ytest=train_test_split(X3,Y1,shuffle=True,stratify=True)
xtrain,xtest,ytrain,ytest=train_test_split(X3,Y2,shuffle=True,stratify=True)
xtrain,xtest,ytrain,ytest=train_test_split(X3,Y3,shuffle=True,stratify=True)



xtrain,xtest,ytrain,ytest=train_test_split(X3,Y1,shuffle=True,stratify=False)
xtrain,xtest,ytrain,ytest=train_test_split(X3,Y3,shuffle=False,stratify=False)

xtrain,xtest,ytrain,ytest=train_test_split(X2,Y2,shuffle=False,stratify=True)
xtrain,xtest,ytrain,ytest=train_test_split(X2,Y3,shuffle=True,stratify=True)


xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X3,Y3,train_size=0.7,test_size=0.2,val_size=0.1,val_split=True,shuffle=True,stratify=True)
xtrain,xtest,xval,ytrain,ytest,yval=train_test_split(X3,Y3,train_size=7,test_size=2,val_size=1,val_split=True,shuffle=True,stratify=False)

