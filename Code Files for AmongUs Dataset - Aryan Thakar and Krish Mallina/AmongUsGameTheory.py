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
# NOTE TO SELF MAKE ARYAN REDO BAR GRAPH FOR CREWMATE TIME TO COMPLETE ALL TASKS

#fills in gaps and organizes the data
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


#normalizing the sabotages fixed column
crewTrainX[:, 5] = (crewTrainX[:, 5] - crewMeans[5]) / crewDenoms[5]
#normalizing the tasks completed column
crewTrainX[:, 0] = (crewTrainX[:, 0] - crewMeans[0]) / crewDenoms[0]
#standardizing the crewmate game length
crewTrainX[:, 3] = (crewTrainX[:, 3] - crewMeans[3]) / crewStandardDevs[3]
#standardizing the crewmate time to complete all tasks
crewTrainX[:, 6] = (crewTrainX[:, 6] - crewMeans[6]) / crewStandardDevs[6]

#standardizing the imposter kills
impTrainX[:, 0] = (impTrainX[:, 0] - impMeans[0]) / impStandardDevs[0]
#standardizing the imposter game length
impTrainX[:, 1] = (impTrainX[:, 1] - impMeans[1]) / impStandardDevs[1]


#normalizing the sabotages fixed column for test data
crewTestX[:, 5] = (crewTestX[:, 5] - crewMeans[5]) / crewDenoms[5]
#normalizing the tasks completed column for test data
crewTestX[:, 0] = (crewTestX[:, 0] - crewMeans[0]) / crewDenoms[0]
#standardizing the crewmate game length for test data
crewTestX[:, 3] = (crewTestX[:, 3] - crewMeans[3]) / crewStandardDevs[3]
#standardizing the crewmate time to complete all tasks for test data
crewTestX[:, 6] = (crewTestX[:, 6] - crewMeans[6]) / crewStandardDevs[6]

#standardizing the imposter kills for test data
impTestX[:, 0] = (impTestX[:, 0] - impMeans[0]) / impStandardDevs[0]
#standardizing the imposter game length for test data
impTestX[:, 1] = (impTestX[:, 1] - impMeans[1]) / impStandardDevs[1]


'''
np.savetxt("CrewmateGameLength.txt", crewmateGameLength)
np.savetxt("ImposterGameLength.txt", imposterGameLength)
np.savetxt("CrewmateTimeToCompleteAllTasks.txt", crewmateTimeCompleteTasks)
np.savetxt("ImposterKills.txt", imposterData[:, 1].astype(np.float))
np.savetxt("SabotagesFixed.txt", crewmateData[:, 6].astype(np.float))
np.savetxt("TasksCompleted.txt", crewmateData[:, 1].astype(np.float))
'''

from keras import models
from keras import layers



"""
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(7, activation='relu'))
"""

modelCrew = models.Sequential()
#input layer, 9 sig fields
modelCrew.add(layers.Dense(720, activation='relu', input_shape=(1408,9) ))
#hidden layer, 2/3 of # of fields in input layer
modelCrew.add(layers.Dense(120, activation='relu'))
#sigmoid/softmax output layer, outputs probability
modelCrew.add(layers.Dense(1, activation='sigmoid'))



from keras import optimizers

modelCrew.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])


historyCrew = modelCrew.fit(crewTrainX, crewTrainY, epochs=48, batch_size=11)
#crewWeights = modelCrew.get_layer('dense_3').get_weights()

modelImp = models.Sequential()

#input layer, 9 sig fields
modelImp.add(layers.Dense(240, activation='relu', input_shape = (372,5) ))
#hidden layer, 2/3 of # of fields in input layer
modelImp.add(layers.Dense(40, activation='relu'))
#sigmoid/softmax output layer, outputs probability

modelImp.add(layers.Dense(1, activation='sigmoid'))

modelImp.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])



historyImp = modelImp.fit(impTrainX, impTrainY, epochs=16, batch_size=36)

"""
derivativ_acc = []
acc = history.history['accuracy']
if (len(acc) > 1):    
    derivativ_acc.append(acc[len(acc) - 1] - acc[len(acc) - 2])
#results = model.evaluate(crewTrainX, crewTrainY)
"""

#results = model.evaluate( crewTrainX , crewTrainY )
print("\n")

for layer in modelImp.layers:
    weightsImp = layer.get_weights() # list of numpy arrays

resultsCrew = modelCrew.evaluate( crewTestX, crewTestY )
resultsImp = modelImp.evaluate( impTestX , impTestY )

predictedCrew = modelCrew.predict(crewTestX)
predictedImp = modelImp.predict(impTestX)

from sklearn import metrics

confusionMatrixCrew = metrics.confusion_matrix(crewTestY, np.rint(predictedCrew))
confusionMatrixImp = metrics.confusion_matrix(impTestY, np.rint(predictedImp))

print ('confusion matrix crewmate:', confusionMatrixCrew)
print ('confusion matrix impostor:', confusionMatrixImp)

#crewWeights = modelCrew.get_weights()
#crewWeights = history.historyImp['weights']
#impWeights= modelImp.get_weights()

