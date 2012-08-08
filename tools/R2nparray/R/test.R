library(Ecdat)

data(Accident)

R2nparray(Accident, fname='accident.txt')

matrix<-replicate(10, rnorm(20)) 


R2nparray(matrix, fname='matrix.txt')

#Try x<-scan() and then enter some numbers

R2nparray(x, fname='scan.txt')