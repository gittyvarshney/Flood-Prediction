import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime

#temp file creation
temp_name = str(input("Please Specify Temperature area: "))
rainfall_name = str(input("Please Specify rainfall area: "))
gdp_per = float(input("Please specify gdp percentage of the state: "))
co2_per = float(input("Please specify the co2 percentage of the state: "))


mydateparser = lambda x: pd.datetime.strptime(x, "%m/%d/%Y")
dft = pd.read_csv("india_temp.csv", usecols=['Date','Temperature','State'], parse_dates=['Date'], date_parser=mydateparser)
arrt = np.array(dft)
arrt[:,0] = pd.to_datetime(arrt[:,0])
print("Starting file temp")
listt1 = []
si = 0
while si < len(arrt):
    if arrt[si][2] == temp_name:
        listt1.append([arrt[si][0],arrt[si][1],arrt[si][2]])
    si +=1
listt1.insert(0,["Date","Temperature","State"])
name_of_file_t = temp_name + '_temp_d.csv'
csvData = listt1
with open(name_of_file_t, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()
print("file " + name_of_file_t + " processed")

#rainfall file creation

dfr = pd.read_csv("india_rainfall.csv", usecols=['AREA','YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
arrr = np.array(dfr)
print("starting file rainfall")

listt2 = []
si = 0

while si < len(arrr):
    if arrr[si][0] == rainfall_name:
        listt2.append([arrr[si][0],arrr[si][1],arrr[si][2],arrr[si][3],arrr[si][4],arrr[si][5],arrr[si][6],arrr[si][7],arrr[si][8],arrr[si][9],arrr[si][10],arrr[si][11],arrr[si][12],arrr[si][13]])
    si +=1

listt2.insert(0,['AREA','YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
name_of_file_r = rainfall_name + '_rainfall_d.csv'
csvData = listt2
with open(name_of_file_r, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()
print("file " + name_of_file_r + " processed")

# reading the created temperature file
mydateparser = lambda x: pd.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
df = pd.read_csv(name_of_file_t, usecols=['Date','Temperature','State'], parse_dates=['Date'], date_parser=mydateparser)
arr = np.array(df)
arr[:,0] = pd.to_datetime(arr[:,0])
print(arr[:20])

# reading the created rainfall file
df2 = pd.read_csv(name_of_file_r, usecols=['AREA','YEAR','JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
arr2 = np.array(df2)
print(arr2[:20])

# reading CO2 file
df3 = pd.read_csv("india_co2.csv", usecols=['CO2 emission'])
arr3 = np.array(df3)
print(arr3[:20])

listt = []
i=0
j=0
while(i < 624):
    listt.append([arr[i][0], arr[i][1], arr3[j][0]*co2_per*0.0001*0.0833, arr2[j][(i%12)+2]])
    if (i+1)%12 == 0:
        j +=1
    i+=1
for i in listt:
    print(i,end="\n")
listt.insert(0,["YEAR","TEMPERATURE","CO2","RAINFALL"])

name = temp_name + '_mnd.csv'

csvData = listt
with open(name, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()
print("file " + name + " processed")



