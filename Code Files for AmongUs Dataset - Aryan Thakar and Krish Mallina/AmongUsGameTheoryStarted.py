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

crewmateData = readData("CrewmateData.csv", 3, 13, 1)
imposterData = readData("ImposterData.csv", 3, 10)
gameLength = crewmateData[:, 4]
test = re.findall("\d+", "07m 24s")

#b = len(crewmateData)


#if 'female', change value to 1; if 'male', change value to 0
#for i in range(len(crewmateData)):
#        crewmateData[i,9][crewmateData[i,9].__contains__('Europe')] = 0

#crewmateData[:,9][crewmateData[:,2]=="male"] = 0

"""  
names = []
marks = []

marks = list( range(10) )   

f = open('CrewmateGameLength.txt','r')
for row in f:
        row = row.split(' ')
        names.append(row[0])
        marks.append(int(marks[0]))
        
plt.bar(names, marks, color = 'g', label = 'File Data')
  
plt.xlabel('Student Names', fontsize = 12)
plt.ylabel('Marks', fontsize = 12)
  
plt.title('Students Marks', fontsize = 20)
plt.legend()
plt.show()
"""

#loads up text file of values in specific significant field 
y = np.loadtxt('CrewmateTimeToCompleteAllTasks.txt', delimiter=' ', unpack=True)


maxNum = np.amax(y)
minNum = np.amin(y)

#makes all the values below 100 into 0
y[:][y[:] <= 100] = 0

#basically divides the numbers by 100, then deletes the remainder. For example, if input is 1714, output is 17.
for i in range(17):
        y[:][ y[:] > (17 - i)*100 ] = 17-i  

 
#creates array of frequencies of each number, such as a 3 appearing 60 times. 
print( np.where( y==maxNum ) )
(unique, counts) = np.unique(y, return_counts=True)
frequencies = np.asarray((unique, counts)).T

#uses this frequency array to create bar chart, to show whether to normalize/standardize/OHE the data. 
plt.bar(frequencies[:,0], frequencies[:,1], color="blue")
plt.title('Time to Complete All Tasks Frequencies')
plt.xlabel("Range")
plt.ylabel("Frequencies")




"""
y2 = np.loadtxt('ImposterGameLength.txt', delimiter=' ', unpack=True)

y_sorted2 = sorted(y2, key = float)
maxNum2 = np.amax(y2)
minNum2 = np.amin(y2)

y2[:][y2[:] <= 100] = 0

for i in range(16):
        y2[:][ y2[:] >= (16 - i)*100 ] = 16-i  
       
print( np.where( y2==0 ) )

(unique2, counts2) = np.unique(y2, return_counts=True)
frequencies2 = np.asarray((unique2, counts2)).T
#x = list( range(1761) ) 


plt.bar(frequencies2[:,0], frequencies2[:,1], color="blue")
plt.title('Impostor Game Length Frequencies')
plt.xlabel("Range")
plt.ylabel("Frequencies")
"""

'''
plt.bar(x, y)
plt.title('Line Graph using NUMPY')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''


