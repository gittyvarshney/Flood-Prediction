# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 23:45:22 2020

@author: pv666
"""
from math import sqrt
import pandas as pd
#from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv


names = ["Begusarai","Saharsa","Kishanpur","Kathihar","Aalo","Pasighat","Kongon","Jiribam","Dhemaji","Sivasagar","Lakhimpur","Nagaon","Bardang","Panchhipara","Midnapore","Jenapur","Cuttack","Purushottampur","Rajahmundry","Vijayawada","Minagallu","Musiri","Sholavandan","Perumbavoor","Vadapuram","Hammigi","Bilgi","Kolhapur","Sangli","Kathapur","Bharuch","Malwadi","Ahmedabad","Aburoad","Kota","Bhanpura","Champa","Poradih","Ranchi","Prayagraj","Faizabad","Moradabad","Delhi","Srinagaru","Rishikesh","Karnal","Bhuntar","Phillaur","Sopore"]
names.sort()
print(names,sep='\n')
listt_file = []
k=0
sr=1
while(k<len(names)):
    variable = names[k] + '.csv'
    file_index = pd.read_csv(variable, usecols = [4])
    file = np.array(file_index)
    #print(file)
    avg = 0.0
    i=0
    while(i<len(file)):
        avg+=file[i][0]
        i+=1
    total = avg/len(file)
    
    variable2 = names[k] + '2.csv'
    file_index2 = pd.read_csv(variable2, usecols = [4])
    file2 = np.array(file_index2)
    avg2 = 0.0
    j=0
    while(j<len(file2)):
        avg2+=file2[j][0]
        j+=1
    total2 = avg2/len(file2)
    difference = total2 - total
    listt_file.append([sr,variable,difference])
    k+=1
    sr+=1
print(listt_file)
name = 'areas_data' + '.csv'
listt_file.insert(0,["Index","Area_name","Diff_Altitude"])

csvData = listt_file
with open(name, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvData)
csvFile.close()
print("file " + name + " processed")

    