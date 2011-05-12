#Kaplan-Meier Estimator
import numpy as np
from math import sqrt

#def a function to calculate the Kaplan-Meier estimate and an estimate of its
#variance

def km(dataArray, timesIn, groupIn, censoringIn):
    #split the data into groups based on the predicting variable
    #get a set of all the groups
    groups = list(set(dataArray[:,groupIn]))
    #create an empty list to store the data for different groups
    groupList = []
    #create an empty list for each group and add it to groups
    for i in range(len(groups)):
        groupList.append([])
    #iterate through all the groups in groups
    for i in range(len(groups)):
       #iterate though the rows of dataArray
        for j in range(len(dataArray)):
            #test if this row has the correct group
            if dataArray[j,groupIn] == groups[i]:
                #add the row to groupList
                groupList[i].append(dataArray[j])
    #create an empty list to store the times for each group
    timeList = []
    #iterate through all the groups
    for i in range(len(groupList)):
        #create an empty list
        times = []
        #iterate through all the rows of the group
        for j in range(len(groupList[i])):
            #get a list of all the times in the group
            times.append(groupList[i][j][timesIn])
        #get a sorted set of the times and store it in timeList
        times = list(sorted(set(times)))
        timeList.append(times)
    #get a list of the number at risk and events at each time
    #create an empty list to store the results in
    timeCounts = []
    #iterate trough each group
    for i in range(len(groupList)):
        #initialize a variable to estimate the survival function
        survival = 1
        #initialize a variable to estimate the variance of the survival function
        varSum = 0
        #initialize a counter for the number at risk
        riskCounter = len(groupList[i])
        #create a list for the counts for this group
        counts = []
        #iterate through the list of times
        for j in range(len(timeList[i])):
            if j != 0:
                #add the count to the list
                counts.append([timeList[i][j-1],riskCounter,eventCounter,survival,sqrt(((survival)**2)*varSum)])
                #increment the number at risk
                riskCounter += -1*(riskChange)
            #initialize a counter for the change in the number at risk
            riskChange = 0
            #initialize a counter to zero
            eventCounter = 0
            #intialize a counter to tell when censoring occurs
            censoringCounter = 0
            #iterate through the observations in each group
            for k in range(len(groupList[i])):
                #check of the observation has the given time
                if (groupList[i][k][timesIn]) == (timeList[i][j]):
                    #increment the number at risk counter
                    riskChange += 1
                    #check if this is an event or censoring
                    if groupList[i][k][censoringIn] == 1:
                        #add 1 to the counter
                        eventCounter += 1
            #check if there are any events at this time
            if eventCounter != censoringCounter:
                censoringCounter = eventCounter
                #calculate the estimate of the survival function
                survival *= ((float(riskCounter) - eventCounter)/(riskCounter))
                try:
                    #calculate the estimate of the variance
                    varSum += (eventCounter)/((riskCounter)*(float(riskCounter)-eventCounter))
                except ZeroDivisionError:
                    varSum = 0
        #append the last row to counts
        counts.append([timeList[i][len(timeList[i])-1],riskCounter,eventCounter,survival,sqrt(((survival)**2)*varSum)])
        #add the list for the group to al ist for all the groups
        timeCounts.append(np.array(counts))
    #returns a list of arrays, where each array has as it columns: the time,
    #the number at risk, the number of events, the estimated value of the
    #survival function at that time, and the estimated standard error at that
    #time, in that order
    return timeCounts
