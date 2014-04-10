#a class for the Kaplan-Meier estimator
from statsmodels.compat.python import range
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

class KAPLAN_MEIER(object):
    def __init__(self, data, timesIn, groupIn, censoringIn):
        raise RuntimeError('Newer version of Kaplan-Meier class available in survival2.py')
        #store the inputs
        self.data = data
        self.timesIn = timesIn
        self.groupIn = groupIn
        self.censoringIn = censoringIn

    def fit(self):
        #split the data into groups based on the predicting variable
        #get a set of all the groups
        groups = list(set(self.data[:,self.groupIn]))
        #create an empty list to store the data for different groups
        groupList = []
        #create an empty list for each group and add it to groups
        for i in range(len(groups)):
            groupList.append([])
        #iterate through all the groups in groups
        for i in range(len(groups)):
           #iterate though the rows of dataArray
            for j in range(len(self.data)):
                #test if this row has the correct group
                if self.data[j,self.groupIn] == groups[i]:
                    #add the row to groupList
                    groupList[i].append(self.data[j])
        #create an empty list to store the times for each group
        timeList = []
        #iterate through all the groups
        for i in range(len(groupList)):
            #create an empty list
            times = []
            #iterate through all the rows of the group
            for j in range(len(groupList[i])):
                #get a list of all the times in the group
                times.append(groupList[i][j][self.timesIn])
            #get a sorted set of the times and store it in timeList
            times = list(sorted(set(times)))
            timeList.append(times)
        #get a list of the number at risk and events at each time
        #create an empty list to store the results in
        timeCounts = []
        #create an empty list to hold points for plotting
        points = []
        #create a list for points where censoring occurs
        censoredPoints = []
        #iterate trough each group
        for i in range(len(groupList)):
            #initialize a variable to estimate the survival function
            survival = 1
            #initialize a variable to estimate the variance of
            #the survival function
            varSum = 0
            #initialize a counter for the number at risk
            riskCounter = len(groupList[i])
            #create a list for the counts for this group
            counts = []
            ##create a list for points to plot
            x = []
            y = []
            #iterate through the list of times
            for j in range(len(timeList[i])):
                if j != 0:
                    if j == 1:
                        #add an indicator to tell if the time
                        #starts a new group
                        groupInd = 1
                        #add (0,1) to the list of points
                        x.append(0)
                        y.append(1)
                        #add the point time to the right of that
                        x.append(timeList[i][j-1])
                        y.append(1)
                        #add the point below that at survival
                        x.append(timeList[i][j-1])
                        y.append(survival)
                        #add the survival to y
                        y.append(survival)
                    else:
                        groupInd = 0
                        #add survival twice to y
                        y.append(survival)
                        y.append(survival)
                        #add the time twice to x
                        x.append(timeList[i][j-1])
                        x.append(timeList[i][j-1])
                    #add each censored time, number of censorings and
                    #its survival to censoredPoints
                    censoredPoints.append([timeList[i][j-1],
                                           censoringNum,survival,groupInd])
                    #add the count to the list
                    counts.append([timeList[i][j-1],riskCounter,
                                   eventCounter,survival,
                                   sqrt(((survival)**2)*varSum)])
                    #increment the number at risk
                    riskCounter += -1*(riskChange)
                #initialize a counter for the change in the number at risk
                riskChange = 0
                #initialize a counter to zero
                eventCounter = 0
                #intialize a counter to tell when censoring occurs
                censoringCounter = 0
                censoringNum = 0
                #iterate through the observations in each group
                for k in range(len(groupList[i])):
                    #check of the observation has the given time
                    if (groupList[i][k][self.timesIn]) == (timeList[i][j]):
                        #increment the number at risk counter
                        riskChange += 1
                        #check if this is an event or censoring
                        if groupList[i][k][self.censoringIn] == 1:
                            #add 1 to the counter
                            eventCounter += 1
                        else:
                            censoringNum += 1
                #check if there are any events at this time
                if eventCounter != censoringCounter:
                    censoringCounter = eventCounter
                    #calculate the estimate of the survival function
                    survival *= ((float(riskCounter) -
                                  eventCounter)/(riskCounter))
                    try:
                        #calculate the estimate of the variance
                        varSum += (eventCounter)/((riskCounter)
                                                  *(float(riskCounter)-
                                                    eventCounter))
                    except ZeroDivisionError:
                        varSum = 0
            #append the last row to counts
            counts.append([timeList[i][len(timeList[i])-1],
                           riskCounter,eventCounter,survival,
                           sqrt(((survival)**2)*varSum)])
            #add the last time once to x
            x.append(timeList[i][len(timeList[i])-1])
            x.append(timeList[i][len(timeList[i])-1])
            #add the last survival twice to y
            y.append(survival)
            #y.append(survival)
            censoredPoints.append([timeList[i][len(timeList[i])-1],
                                   censoringNum,survival,1])
            #add the list for the group to al ist for all the groups
            timeCounts.append(np.array(counts))
            points.append([x,y])
        #returns a list of arrays, where each array has as it columns: the time,
        #the number at risk, the number of events, the estimated value of the
        #survival function at that time, and the estimated standard error at
        #that time, in that order
        self.results = timeCounts
        self.points = points
        self.censoredPoints = censoredPoints

    def plot(self):
        x = []
        #iterate through the groups
        for i in range(len(self.points)):
            #plot x and y
            plt.plot(np.array(self.points[i][0]),np.array(self.points[i][1]))
            #create lists of all the x and y values
            x += self.points[i][0]
        for j in range(len(self.censoredPoints)):
            #check if censoring is occuring
            if (self.censoredPoints[j][1] != 0):
                #if this is the first censored point
                if (self.censoredPoints[j][3] == 1) and (j == 0):
                    #calculate a distance beyond 1 to place it
                    #so all the points will fit
                    dx = ((1./((self.censoredPoints[j][1])+1.))
                    *(float(self.censoredPoints[j][0])))
                    #iterate through all the censored points at this time
                    for k in range(self.censoredPoints[j][1]):
                        #plot a vertical line for censoring
                        plt.vlines((1+((k+1)*dx)),
                                   self.censoredPoints[j][2]-0.03,
                                   self.censoredPoints[j][2]+0.03)
                #if this censored point starts a new group
                elif ((self.censoredPoints[j][3] == 1) and
                      (self.censoredPoints[j-1][3] == 1)):
                    #calculate a distance beyond 1 to place it
                    #so all the points will fit
                    dx = ((1./((self.censoredPoints[j][1])+1.))
                    *(float(self.censoredPoints[j][0])))
                    #iterate through all the censored points at this time
                    for k in range(self.censoredPoints[j][1]):
                        #plot a vertical line for censoring
                        plt.vlines((1+((k+1)*dx)),
                                   self.censoredPoints[j][2]-0.03,
                                   self.censoredPoints[j][2]+0.03)
                #if this is the last censored point
                elif j == (len(self.censoredPoints) - 1):
                    #calculate a distance beyond the previous time
                    #so that all the points will fit
                    dx = ((1./((self.censoredPoints[j][1])+1.))
                    *(float(self.censoredPoints[j][0])))
                    #iterate through all the points at this time
                    for k in range(self.censoredPoints[j][1]):
                        #plot a vertical line for censoring
                        plt.vlines((self.censoredPoints[j-1][0]+((k+1)*dx)),
                                   self.censoredPoints[j][2]-0.03,
                                   self.censoredPoints[j][2]+0.03)
                #if this is a point in the middle of the group
                else:
                    #calcuate a distance beyond the current time
                    #to place the point, so they all fit
                    dx = ((1./((self.censoredPoints[j][1])+1.))
                    *(float(self.censoredPoints[j+1][0])
                      - self.censoredPoints[j][0]))
                    #iterate through all the points at this time
                    for k in range(self.censoredPoints[j][1]):
                        #plot a vetical line for censoring
                        plt.vlines((self.censoredPoints[j][0]+((k+1)*dx)),
                                   self.censoredPoints[j][2]-0.03,
                                   self.censoredPoints[j][2]+0.03)
        #set the size of the plot so it extends to the max x and above 1 for y
        plt.xlim((0,np.max(x)))
        plt.ylim((0,1.05))
        #label the axes
        plt.xlabel('time')
        plt.ylabel('survival')
        plt.show()

    def show_results(self):
        #start a string that will be a table of the results
        resultsString = ''
        #iterate through all the groups
        for i in range(len(self.results)):
            #label the group and header
            resultsString += ('Group {0}\n\n'.format(i) +
            'Time     At Risk     Events     Survival     Std. Err\n')
            for j in self.results[i]:
                #add the results to the string
                resultsString += (
                '{0:<9d}{1:<12d}{2:<11d}{3:<13.4f}{4:<6.4f}\n'.format(
                    int(j[0]),int(j[1]),int(j[2]),j[3],j[4]))
        print(resultsString)
