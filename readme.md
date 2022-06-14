# Flood Forecasting
We See that the ongoing techniques to forecast floods can only extends their power 
to forecast the upcoming floods to only a week, hence the time of preparing for the 
upcoming disaster reduced which incase affects the evacuation and other necessary 
relief efforts, also the previous meteorological methods to forecast the floods do 
not take into consideration the previous datasets i.e. temperature, Carbon emission 
which may play a huge role in forecasting the heavy rainfall. Also by observing the 
recent calamities we can say that the standard forecasting methods are not enough 
to predict such complex type of disaster which takes into consideration various 
factors.

![Flood-Image](https://github.com/gittyvarshney/Flood-Prediction/blob/main/flood_image.png?raw=true)

## Idea
-   Now a days Machine learning and Data Science which is emerging as a Key player 
in computation can give us answers to many problems. 
-   We focused on the problem 
through an approach of Geo referenced data mining and machine learning. 
-   The 
Idea is to estimate the upcoming floods of the next year through the mechanism of 
LSTM 
-   We predict the rainfall patterns of the next year by taking into account the values of temperature and carbon emission, so the estimate can also include the climate change patterns then including the Geo Referenced data to 
provide clearer picture.

## DataSet Used
-   TEMPERATURE PER MONTH PER STATE(1961-2012)
-   RAINFALL PER MONTH PER ZONE (1961-2012)
-   CO2 EMISSION ANNUAL (1961-2012)
-   GEO REFERENCED ALTITUDES OF LOW LYING AREAS
-   RIVER WATER LEVELS

The above Dataset first been converted to readable and organized form by using the `DataState` > `state_wise_minning.py` (already been converted generated)

## Geo Mining
We have mined total 49 areas throughout India alongside the rivers which have the 
high probability of floods due to proximity to flooding rivers and low lying areas. 
So In order to avoid the complexity of contour lines we have used only two 
concentric lines to get the idea about the difference in the altitudes of a particular 
low-lying area <br /> <br />
For example the areas of Jenapur along side the Mahanadi River looked something 
like:

![Jenapur](https://github.com/gittyvarshney/Flood-Prediction/blob/main/geo_pic.jpg?raw=true)

For all 49 areas It will look something like: <br />
![Full India](https://github.com/gittyvarshney/Flood-Prediction/blob/main/flood_prediction_bg.png?raw=true)

After Drawing two concentric spatial lines for each area we save these files to Kml 
and add altitudes to these files with the help GPS Visualizer. Hence the gpx file became something like:
![GPX File](https://github.com/gittyvarshney/Flood-Prediction/blob/master/gpx_file.png?raw=true)
and converting them to csv format with the help of TCX Converter 

## LSTM Implementation
Humans Do prediction on the basis of the Logic power and memory power same 
can be simulated on artificial intelligence through the use of newly invented neural 
networks which generate results by various activation functions (as logic) and feed 
forward or backward (as memory), but the neural networks suffers from the 
problems of vanishing gradient and Exploding gradient due to which LSTM was 
devised. 

-    LSTM basically works on the mechanism of Feeding the Memory of the 
previous results and generating the new Memory through some specified 
parameters
-   How much the current output is affected by the previous outputs and 
previous memory.

![LSTM](https://github.com/gittyvarshney/Flood-Prediction/blob/main/lstm.png?raw=true)
<br/>

We have taken previous time memory feed for a row upto a time of (t-4) 

### LSTM Overview
-   Normalize the dataset in range of (0,1) as LSTM works with tanh and
sigmoid activation logic.
-   Splitting the Data into train and test datasets
-   Reshaping the Data to be fit into LSTM
-   Creating a LSTM Network with four neural networks.
-   Training the Data for upto 72 epochs to minimize the error
-    Predicting on Test datasets

### Increase In Water Levels
Now We have predicted rainfalls so inorder to predict how much the water level as 
risen due to the rainfall we choose a factor of 0.012 which is devised in a paper in 
forecasting water levels in Pahang river.

### Flood Categoriztion
Now as we have the increment in river levels we can deduce if that increase may 
lead the river level higher than the flood levels if so then the flood may occur 
which in turn depends upon the altitude of the surrounding areas, if the water rises 
upto the difference of 5 meters then the floods can be severe and so on.

### Making Shape File
In order to analyze the results of different areas we have to visualize it onto a map 
so as to get a idea which part of the country may me affected by floods for this we 
use the ArcGis Software where we generated our custom shape file of India. 
India Boundary & Flood Areas (India_boundary.shp & flood_areas.shp) <br /> (already been Done !)
![India Map](https://github.com/gittyvarshney/Flood-Prediction/blob/main/india_map.png?raw=true)

## Steps To Reproduce Results
-   Open lstm_implement.py in spyder or any other editor and run it

However, If you want to change to some other area 
-   First Highlight the area with two layers in Google Earth
-   Generate corresponding kml files of the (Lat/Long)
-   Get altitudes from the kml files using any online [GPS Visualizer](https://www.gpsvisualizer.com/)
-  convert the generated gpx file to csv using [TCX Convertor](https://tcx-converter.software.informer.com/2.0/)
-   Take average of altitudes and deduce corresponding difference between them
-   Get Normal river levels and flood river levels in your area from [WRIS](https://indiawris.gov.in/wris/#/RiverMonitoring)
-   Mention the corresponding details in areas_data.csv and also mention the state_index (refer to `file_destination.csv`)
-   Optional, If you want to visualize edit the `digital` > `flood_areas.shx` using [ArcGIS](https://www.esri.com/en-us/arcgis/products/arcgis-desktop/resources) 




