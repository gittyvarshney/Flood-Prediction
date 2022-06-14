from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import geopandas as gpd
import pandas as pd
#from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import csv

file_path = pd.read_csv("files_destination.csv", usecols = [0,1])
print(file_path.head())
arr_path= np.array(file_path)

state_rainfall = []
lstm_results = []

for n in range(len(arr_path)): # len(arr_path)
    print("Currently executing: ",arr_path[n][0])
    variable = str(arr_path[n][0])
    datasetset = read_csv(variable, header=0, index_col=0)
    print(datasetset.head())
    
    values = datasetset.values
    average_rainfall = np.array(values)
    values = values.astype('float32')
    print(values)
    
    #geeting averages of rainfall in months of july, august and september months
    i=0
    l=0
    k=0
    rain_mm = []
    while(i < 588):
        if (i)%12 == 0:
            l =0
            k +=1
        if l== 6 or l== 7 or l==8:
            rain_mm.append(float(average_rainfall[i][2]))
            #print(average_rainfall[i][3])
    
        i+=1
        l+=1
    k = k-1
    i = 0
    #print(average_rainfall)
    total = sum(rain_mm)
    average_rain = total/(k*3)
    print(average_rain)
    #print(values)
    #print(rain_mm)
    
    scaler_data = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler_data.fit_transform(values)
    print(scaled_data)
    inverse_transform_y = scaler_data.inverse_transform(scaled_data)
    #print(inverse_transform_y)
    
    #Multivariable supervision (t-4) (t-3) (t-2) (t-1) t
    def building_supervision_series(dataset, n_in=1, n_out=1, dropnan=True):
    	n_vars = 1 if type(dataset) is list else dataset.shape[1]
    	df = DataFrame(dataset)
    	columns, nomenclature = list(), list()
    
    	for i in range(n_in, 0, -1):
    		columns.append(df.shift(i))
    		nomenclature += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    	
    	for i in range(0, n_out):
    		columns.append(df.shift(-i))
    		if i == 0:
    			nomenclature += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    		else:
    			nomenclature += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    	
    	aggregate_series = concat(columns, axis=1)
    	aggregate_series.columns = nomenclature
    	
    	if dropnan:
    		aggregate_series.dropna(inplace=True)
    	return aggregate_series
    
    #function to execute multivariable execution
    reframed = building_supervision_series(scaled_data, 4, 1)
    print(reframed[0:20])
    
    reframed.drop(reframed.columns[[12,13]], axis=1, inplace=True)
    print(reframed[:20])
    
    #Not taking into consideration the previous values of the CO2 (t-4), (t-3), (t-2), (t-1), t
    values = reframed.values
    for i in range(len(values)):
      values[i][1] = 0 
      values[i][4] = 0 
      values[i][7] = 0
      
    print(values[:20])
    
    values = reframed.values
    n_train = 584
    train = values[:n_train, :]
    test = values[n_train:, :]
    
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    
    print(train_X.shape[0])
    
    train_X = train_X.reshape((train_X.shape[0], 4, 3))
    print(train_X[:20])
    
    
    test_X = test_X.reshape((test_X.shape[0], 4, 3))
    #print("printing test_X\n")
    #print(test_X[:20])
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    
    history = model.fit(train_X, train_y, epochs=72, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    
    #pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    #pyplot.legend()
    #pyplot.show()
    
    
    #Deducing the RMSE From the training model
    predicted_y = model.predict(train_X)
    print(predicted_y)
    
    #print(train_X)
    train_X = train_X.reshape((train_X.shape[0], 12))
    inverse_transform_y = concatenate((train_X[:,9:11],predicted_y), axis=1)
    
    inverse_transform_y = scaler_data.inverse_transform(inverse_transform_y)
    print(inverse_transform_y)
    print("\nafter\n")
    inverse_transform_y = inverse_transform_y[:,-1]
    print(inverse_transform_y)
    
    train_y = train_y.reshape((len(train_y), 1))
    actual_y = concatenate((train_X[:, 9:11],train_y), axis=1)
    actual_y = scaler_data.inverse_transform(actual_y)
    actual_y = actual_y[:,-1]
    train_rmse = sqrt(mean_squared_error(actual_y, inverse_transform_y))
    print(train_rmse)
    
    
    predicted_y = model.predict(test_X)
    #print(predicted_y)
    print("test_X\n")
    #print(test_X)
    
    test_X = test_X.reshape((test_X.shape[0], 12))
    inverse_transform_y = concatenate((test_X[:,9:11],predicted_y), axis=1)
    
    inverse_transform_y = scaler_data.inverse_transform(inverse_transform_y)
    inverse_transform_y = inverse_transform_y[:,-1]
    #print(inverse_transform_y)
    
    test_y = test_y.reshape((len(test_y), 1))
    actual_y = concatenate((test_X[:, 9:11],test_y), axis=1)
    actual_y = scaler_data.inverse_transform(actual_y)
    actual_y = actual_y[:,-1]
    #print(actual_y)
    
    pyplot.plot(inverse_transform_y, label='predicted')
    pyplot.plot(actual_y, label='actual')
    pyplot.legend()
    pyplot.show()
    rmse = sqrt(mean_squared_error(actual_y, inverse_transform_y))
    print('Test RMSE: %.3f' % rmse)
    lstm_results.append([n,arr_path[n][0],rmse])
    #print(arr_index)
    
    i=0
    l=0
    k=0
    
    rainfall_extra = []
    while i < len(inverse_transform_y):
        print(inverse_transform_y[i])
        if (i)%12 == 0:
            l =0
            k = 0
        if l== 6 or l== 7 or l==8:
            k += float(inverse_transform_y[i])
        if l == 9:
            k = k*(1+0.01*train_rmse)
            rainfall_extra.append((k/3))
        i+=1
        l+=1
    print('Predicted extra rainfall ',rainfall_extra[0]," ",rainfall_extra[1]," ",rainfall_extra[2])
    state_rainfall.append([n,rainfall_extra[0]*0.012,rainfall_extra[1]*0.012,rainfall_extra[2]*0.012])
state_rainfall.insert(0,["State_index","2010","2011","2012"])
lstm_results.insert(0,["index","state","RMSE"])

name_of_file = 'lstm_results' + '.csv'

csvdataset = lstm_results
with open(name_of_file, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvdataset)
csvFile.close()
print("file " + name_of_file + " processed")


name_of_file = 'increase_in_water_levels' + '.csv'

csvdataset = state_rainfall
with open(name_of_file, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvdataset)
csvFile.close()
print("file " + name_of_file + " processed")

#specifiying flooding conditions
state_rainfall = pd.read_csv("increase_in_water_levels.csv", usecols = [0,1,2,3])
print(state_rainfall.head())
state_rainfall = np.array(state_rainfall)

area_path = pd.read_csv("areas_data.csv", usecols = [0,1,2,3,4,5])
print(area_path.head())
area_path = np.array(area_path)
floods_condition = []

for p in range(len(area_path)):
    listt1 = []
    for q in range(3):
        var=0;
        if area_path[p][3] + state_rainfall[area_path[p][5]][q+1] >= area_path[p][4]:
            if area_path[p][2] <= 0:
                var = 5
            elif area_path[p][2] < 5:
                var = 4
            elif area_path[p][2] < 10:
                var = 3
            elif area_path[p][2] < 15:
                var = 2
            else:
                var = 1
        else:
            var = 0
        listt1.append(var)
    floods_condition.append([area_path[p][1],listt1[0],listt1[1],listt1[2]])
floods_condition.insert(0,["Area_Name","2010","2011","2012"])
print(floods_condition)     

name = 'flood_condition' + '.csv'
csvdataset = floods_condition
with open(name, 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csvdataset)
csvFile.close()

fp = "digital/flood_areas.shp"
map_df = gpd.read_file(fp)

fp2 = "digital\\india_boundary.shp"
map_df2 = gpd.read_file(fp2)

listt_states = ['2010','2011','2012']
df1 = pd.read_csv("floods_condition.csv")
for i in range(len(listt_states)):
    
    #print(df1)
    merged = map_df.set_index('Name').join(df1.set_index('Area_Name'))
    #print(merged.head())
    variable = listt_states[i]
    vmin, vmax = 0,5
    fig, ax = plt.subplots(1,figsize=(10,10))
    merged.plot(column=variable,cmap='Reds',linewidth=0.8,ax=ax,edgecolor='0.8')
    map_df2.plot(ax=ax)
    ax.axis('off')
    title_variable = 'Prediction of Floods ' + variable
    ax.set_title(title_variable, fontdict = {'fontsize': '25','fontweight': '3'})
    ax.annotate('Predictive through LSTM Multivariate ',xy=(0.1,.08),xycoords = 'figure fraction',horizontalalignment='left',verticalalignment='top', fontsize=12, color='#555555')
    sm = plt.cm.ScalarMappable(cmap ='Reds', norm=plt.Normalize(vmin=vmin,vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm)    

