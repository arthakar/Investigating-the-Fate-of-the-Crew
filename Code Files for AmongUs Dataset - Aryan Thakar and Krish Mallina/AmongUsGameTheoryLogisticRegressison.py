# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:44:40 2022

@author: 848223

Source for dataset: https://www.kaggle.com/ruchi798/among-us-dataset

"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import sklearn.model_selection as model_selection
import re

def readData(name, minRange, maxRange, rowskip = 0):
    fileName = name
    raw_data = open(fileName, 'rt')
    
    data = np.loadtxt(raw_data, usecols = np.arange(minRange, maxRange), 
                      skiprows = rowskip, delimiter = ',', dtype=np.str)
    return data

crewmateData = readData("CrewmateData.csv", 3, 11, 1)
crewmateRegion = readData("CrewmateData.csv", 12, 13, 1)
crewmateRegion = np.reshape(crewmateRegion, (len(crewmateRegion), 1))
imposterData = readData("ImposterData.csv", 3, 7, 1)
imposterRegion = readData("ImposterData.csv", 9, 10, 1)
imposterRegion = np.reshape(imposterRegion, (len(imposterRegion), 1))

#puts game length of crewmate data in seconds
gameLength = crewmateData[:, 4]
crewmateGameLength = np.empty((len(gameLength), 1))
for count,value in enumerate(gameLength):
    holder = np.array(re.findall(r"\d+", value))
    holder = holder.astype(np.float)
    crewmateGameLength[count, 0] = holder[0] * 60 + holder[1]
    
#puts game length of imposter data in seconds
gameLength = imposterData[:, 2]
imposterGameLength = np.empty((len(gameLength), 1))
for count,value in enumerate(gameLength):
    holder = np.array(re.findall(r"\d+", value))
    holder = holder.astype(np.float)
    imposterGameLength[count, 0] = holder[0] * 60 + holder[1]
    
#puts time to complete task of crewmates in seconds
taskLength = crewmateData[:, 7]
crewmateTimeCompleteTasks = np.empty((len(taskLength), 1))
for count,value in enumerate(taskLength):
    if value != "-":
        holder = np.array(re.findall(r"\d+", value))
        holder = holder.astype(np.float)
        crewmateTimeCompleteTasks[count, 0] = holder[0] * 60 + holder[1]
    if value == "-":
        crewmateTimeCompleteTasks[count, 0] = np.nan
avg = np.nanmean(crewmateTimeCompleteTasks[:, 0])
crewmateTimeCompleteTasks[:,0][np.isnan(crewmateTimeCompleteTasks[:, 0])] = avg

#one hot encode crewmate game regions
crewmateRegion[:, 0][np.flatnonzero(np.char.find(crewmateRegion, "Europe") + 1)] = 1
crewmateRegion[:, 0][np.flatnonzero(np.char.find(crewmateRegion, "NA") + 1)] = 0
crewmateRegion = crewmateRegion.astype(np.float)
ohe = OneHotEncoder(categories = "auto")
crewmateRegion = ohe.fit_transform(crewmateRegion).toarray().astype(np.float)

#one hot encode imposter game regions
imposterRegion[:, 0][np.flatnonzero(np.char.find(imposterRegion, "Europe") + 1)] = 1
imposterRegion[:, 0][np.flatnonzero(np.char.find(imposterRegion, "NA") + 1)] = 0
imposterRegion = imposterRegion.astype(np.float)
ohe = OneHotEncoder(categories = "auto")
imposterRegion = ohe.fit_transform(imposterRegion).toarray().astype(np.float)

#data formatting for crewmates
outcomes = np.reshape(crewmateData[:, 0], (len(crewmateData), 1))
crewmateData = np.delete(crewmateData, 0, axis = 1)

outcomes[:, 0][outcomes[:, 0] == "Win"] = 1
outcomes[:, 0][outcomes[:, 0] == "Loss"] = 0

crewmateData = np.insert(crewmateData, 3, crewmateGameLength.flatten(), axis = 1)
crewmateData = np.delete(crewmateData, 4, axis = 1)
crewmateData = np.insert(crewmateData, 6, crewmateTimeCompleteTasks.flatten(), axis = 1)
crewmateData = np.delete(crewmateData, 7, axis = 1)

crewmateData[:, 1][crewmateData[:, 1] == "Yes"] = 1
crewmateData[:, 1][crewmateData[:, 1] == "No"] = 0

crewmateData[:, 2][crewmateData[:, 2] == "Yes"] = 1
crewmateData[:, 2][crewmateData[:, 2] == "No"] = 0

crewmateData[:, 4][crewmateData[:, 4] == "Yes"] = 1
crewmateData[:, 4][crewmateData[:, 4] == "No"] = 0

crewmateData = np.append(crewmateData, crewmateRegion, axis = 1)

crewmateData = np.append(crewmateData, outcomes, axis = 1)
crewmateData = crewmateData.astype(np.float)
#crewmateData:
    # 0th col: Tasks Completed
    # 1st col: All Tasks Completed?
    # 2nd col: Murdered?
    # 3rd col: Crewmate Game Length
    # 4th col: Ejected?
    # 5th col: Sabotages Fixed
    # 6th col: Time to Complete All Tasks
    # 7th col: NA Player Region 
    # 8th col: Europe Player Region
    # 9th col: Outcome (Win = 1; Loss = 0)
#data formatting for imposters
outcomes = np.reshape(imposterData[:, 0], (len(imposterData), 1))

outcomes[:, 0][outcomes[:, 0] == "Win"] = 1
outcomes[:, 0][outcomes[:, 0] == "Loss"] = 0

imposterData = np.delete(imposterData, 0, axis = 1)

imposterData = np.insert(imposterData, 1, imposterGameLength.flatten(), axis = 1)
imposterData = np.delete(imposterData, 2, axis = 1)

imposterData[:, 2][imposterData[:, 2] == "Yes"] = 1
imposterData[:, 2][imposterData[:, 2] == "No"] = 0

imposterData = np.append(imposterData, imposterRegion, axis = 1)

imposterData = np.append(imposterData, outcomes, axis = 1)
imposterData = imposterData.astype(np.float)
#imposterData:
    # 0th col: Imposter Kills
    # 1st col: Imposter Game Length
    # 2nd col: Ejected?
    # 3rd col: NA Player Region
    # 4th col: Europe Player Region
    # 5th col: Outcome (Win = 1; Loss = 0)

    
# SEPARATION OF DATA
crewTrainX, crewTestX, crewTrainY, crewTestY = \
    model_selection.train_test_split(crewmateData[:, 0:9], crewmateData[:, 9],
                          train_size=0.80,test_size=0.20,
                          random_state=101)

impTrainX, impTestX, impTrainY, impTestY = \
    model_selection.train_test_split(imposterData[:, 0:5], imposterData[:, 5],
                          train_size=0.80,test_size=0.20,
                          random_state=101)

#gathering constants for the crew data
crewMeans = np.mean(crewTrainX, axis = 0)
crewDenoms = np.amax(crewTrainX, axis = 0) - np.amin(crewTrainX, axis = 0)
crewStandardDevs = np.std(crewTrainX, axis = 0)

#gathering constants for the imp data
impMeans = np.mean(impTrainX, axis = 0)
impStandardDevs = np.std(impTrainX, axis = 0)

#normalizing the sabotages fixed column for test and train data
crewTrainX[:, 5] = (crewTrainX[:, 5] - crewMeans[5]) / crewDenoms[5]
crewTestX[:, 5] = (crewTestX[:, 5] - crewMeans[5]) / crewDenoms[5]

#normalizing the tasks completed column for test and train data
crewTrainX[:, 0] = (crewTrainX[:, 0] - crewMeans[0]) / crewDenoms[0]
crewTestX[:, 0] = (crewTestX[:, 0] - crewMeans[0]) / crewDenoms[0]

#standardizing the crewmate game length for test and train data
crewTrainX[:, 3] = (crewTrainX[:, 3] - crewMeans[3]) / crewStandardDevs[3]
crewTestX[:, 3] = (crewTestX[:, 3] - crewMeans[3]) / crewStandardDevs[3]

#standardizing the crewmate time to complete all tasks for test and train data
crewTrainX[:, 6] = (crewTrainX[:, 6] - crewMeans[6]) / crewStandardDevs[6]
crewTestX[:, 6] = (crewTestX[:, 6] - crewMeans[6]) / crewStandardDevs[6]

#standardizing the imposter kills for train and test data
impTrainX[:, 0] = (impTrainX[:, 0] - impMeans[0]) / impStandardDevs[0]
impTestX[:, 0] = (impTestX[:, 0] - impMeans[0]) / impStandardDevs[0]

#standardizing the imposter game length for test and train data
impTrainX[:, 1] = (impTrainX[:, 1] - impMeans[1]) / impStandardDevs[1]
impTestX[:, 1] = (impTestX[:, 1] - impMeans[1]) / impStandardDevs[1]

#adding bias cols
crewTrainX = np.append(np.ones((len(crewTrainX), 1)), crewTrainX, axis = 1)
crewTestX = np.append(np.ones((len(crewTestX), 1)), crewTestX, axis = 1)
impTrainX = np.append(np.ones((len(impTrainX), 1)), impTrainX, axis = 1)
impTestX = np.append(np.ones((len(impTestX), 1)), impTestX, axis = 1)



creWeights = np.zeros((len(crewTrainX[0]), 1))
# 0th: bias
# 1st: Tasks Completed
# 2nd: All Tasks Completed?
# 3rd: Murdered?
# 4th: Crewmate Game Length
# 5th: Ejected?
# 6th: Sabotages Fixed
# 7th: Time to Complete All Tasks
# 8th: NA Player Region 
# 9th: Europe Player Region

impWeights = np.zeros((len(impTrainX[0]), 1))
# 0th: bias
# 1st: Imposter Kills
# 2nd: Imposter Game Length
# 3rd: Ejected?
# 4th: NA Player Region
# 5th: Europe Player Region

crewTrainY = np.reshape(crewTrainY, (len(crewTrainY), 1))
crewTestY = np.reshape(crewTestY, (len(crewTestY), 1))
impTrainY = np.reshape(impTrainY, (len(impTrainY), 1))
impTestY = np.reshape(impTestY, (len(impTestY), 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def costFunc(x, weights, y):
    #temporary y array because i dont want to affect the actual array
    temp_y = y
    #dots y (1x11) and sigmoid result (11x1) and adds up the ones in the y array
    cost = np.dot(np.transpose(temp_y), np.log(sigmoid(np.dot(x, weights))))
    temp_y = -1 * y + 1
    #dots y (1x11) and 1 - sigmoid result (11x1) and adds up the zeroes in the y array
    cost = cost + np.dot(np.transpose(temp_y), np.log(1 - sigmoid(np.dot(x, weights))))
    cost = cost / len(x)
    cost = cost * -1
    return float(cost)

def grad_Desc(x, weights, y):
    gradients = sigmoid(np.dot(x, weights)) - y
    gradients = gradients / len(x)
    gradients = np.dot(np.transpose(x), gradients)
    return gradients

LR = 1.8
vectNorm = 1
tolerance = 0.000001
costArray = []

while (vectNorm > tolerance):
    cost = costFunc(crewTrainX, creWeights, crewTrainY)
    costArray.append(cost)
    
    gradients = grad_Desc(crewTrainX, creWeights, crewTrainY)
    
    creWeights = creWeights - (LR * gradients)
    vectNorm = np.linalg.norm(gradients)
    
# plot cost
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(len(costArray)), costArray, color='blue')
#ax2.set(title = 'Cost vs. Iterations', xLabel = 'iterations', yLabel = 'cost')

LR = 1.2
vectNorm = 1
tolerance = 0.01
costArray = []

while (vectNorm > tolerance):
    cost = costFunc(impTrainX, impWeights, impTrainY)
    costArray.append(cost)
    
    gradients = grad_Desc(impTrainX, impWeights, impTrainY)
    
    impWeights = impWeights - (LR * gradients)
    vectNorm = np.linalg.norm(gradients)
    
# plot cost
fig2 = plt.figure()
ax2 = fig2.add_axes([0.1, 0.1, 0.8, 0.8])
ax2.plot(range(len(costArray)), costArray, color='red')
#ax2.set(title = 'Cost vs. Iterations', xLabel = 'iterations', yLabel = 'cost')

crewPrediction = sigmoid(np.dot(crewTestX, creWeights))
crewPrediction = np.around(crewPrediction)
impPrediction = sigmoid(np.dot(impTestX, impWeights))
impPrediction = np.around(impPrediction)

# -1s are false positives; 1s are false negatives; 0s are correct predictions
diff = crewTestY - crewPrediction
errorCounter = np.count_nonzero(diff)

crewErrorRate = errorCounter / len(crewTestY)
crewAccuracy = 1 - crewErrorRate

diff = impTestY - impPrediction
errorCounter = np.count_nonzero(diff)

impErrorRate = errorCounter / len(impTestY)
impAccuracy = 1 - impErrorRate

falsePositive = (diff == -1).sum()
falseNegative = (diff == 1).sum()

# need to do this to separate the true negatives from true positives
indices = np.where(diff == 0)
truths = impTestY[indices] - diff[indices]

truePositive = np.count_nonzero(truths)
trueNegative = len(truths) - truePositive

precision = truePositive / (truePositive + falsePositive)
recall = truePositive / (truePositive + falseNegative)

print("%15s: %4f" % ("Accuracy", impAccuracy))
print("%15s: %4f" % ("Error Rate", impErrorRate))
print("%15s: %4f" % ("Precision", precision))
print("%15s: %4f" % ("Recall", recall))
print()
'''
np.savetxt("CrewmateGameLength.txt", crewmateGameLength)
np.savetxt("ImposterGameLength.txt", imposterGameLength)
np.savetxt("CrewmateTimeToCompleteAllTasks.txt", crewmateTimeCompleteTasks)
np.savetxt("ImposterKills.txt", imposterData[:, 1].astype(np.float))
np.savetxt("SabotagesFixed.txt", crewmateData[:, 6].astype(np.float))
np.savetxt("TasksCompleted.txt", crewmateData[:, 1].astype(np.float))
'''