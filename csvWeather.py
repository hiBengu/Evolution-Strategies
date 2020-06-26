import pandas as pd
import numpy as np
import pickle

# Read CSV file
dataFrame = pd.read_csv(
    "city_temperature.csv",      # relative python path to subdirectory
    sep=',',         # Tab-separated value file.
    skiprows=0,         # Skip the first 10 rows of the file
)
print("Reading CSV File is Done!")

dataFrame = dataFrame[['Region','Country','City','Month','Day', 'Year', 'AvgTemperature']]

print("Necessary columns are selected!")

dataFrame['Region'], levelsRegion = pd.factorize(dataFrame['Region'])
print(levelsRegion)

dataFrame = dataFrame[dataFrame['Year'] > 1994]
dataFrame = dataFrame[dataFrame['Year'] < 2019]
dataFrame = dataFrame[dataFrame['Region'] == 3]
print("Values below year 2010 is deleted")

dataFrame['Country'], levelsCountry = pd.factorize(dataFrame['Country'])
print(levelsCountry)
print(len(levelsCountry))
print(min(dataFrame['Country']))
print(max(dataFrame['Country']))
dataFrame['City'], levelsCity = pd.factorize(dataFrame['City'])
print(levelsCity)
print(len(levelsCity))
print(min(dataFrame['City']))
print(max(dataFrame['City']))
print("String values in data frame are converted to int!")
print(min(dataFrame['Month']))
print(min(dataFrame['Day']))

dataFrame = dataFrame.sample(frac=1).reset_index(drop=True)
print("DataFrame is shuffled!")

dataFrame = dataFrame[dataFrame['AvgTemperature'] > -50]
print("Rows with avg temperature below -50 is removed")

dataFrame['AvgTemperature'] =  (dataFrame['AvgTemperature'] - 32) * 5 / 9

dataFrameTrain = dataFrame[dataFrame['Year'] < 2017]
dataFrameTest = dataFrame[dataFrame['Year'] > 2016]

dataFrameTrain['Country'] /= max(dataFrameTrain['Country'])
dataFrameTrain['Month'] /= max(dataFrameTrain['Month'])
dataFrameTrain['Day'] /= max(dataFrameTrain['Day'])
dataFrameTrain['Year'] = dataFrameTrain['Year'] - 1994
print(f"max Value Year: {max(dataFrameTrain['Year'])}")
dataFrameTrain['Year'] = dataFrameTrain['Year']/max(dataFrameTrain['Year'])
dataFrameTrain['Region'] /= max(dataFrameTrain['Region'])
dataFrameTrain['City'] /= max(dataFrameTrain['City'])

dataFrameTest['Country'] /= max(dataFrameTest['Country'])
dataFrameTest['Month'] /= max(dataFrameTest['Month'])
dataFrameTest['Day'] /= max(dataFrameTest['Day'])
dataFrameTest['Year'] = dataFrameTest['Year'] - 1994
dataFrameTest['Year'] /= max(dataFrameTest['Year'])
dataFrameTest['Region'] /= max(dataFrameTest['Region'])
dataFrameTest['City'] /= max(dataFrameTest['City'])
print("Normalization!")
print(dataFrameTrain.head())
print(dataFrameTest)



xDataTrain = dataFrameTrain.iloc[:, 0:6]
yDataTrain = dataFrameTrain.iloc[:, 6]

# xDataTrain = dataFrameTrain.iloc[:300000, 0:6]
# yDataTrain = dataFrameTrain.iloc[:300000, 6]
#
# xDataTest = dataFrameTrain.iloc[300000:, 0:6]
# yDataTest = dataFrameTrain.iloc[300000:, 6]

xDataTest = dataFrameTest.iloc[:, 0:6]
yDataTest = dataFrameTest.iloc[:, 6]

print(xDataTrain.shape)
print(yDataTrain.shape)
print("xData and yData are created!")

print(xDataTest.shape)
print(yDataTest.shape)

print(xDataTest)
print(xDataTrain)

xDataTrain = xDataTrain.to_numpy()
yDataTrain = yDataTrain.to_numpy()
xDataTest = xDataTest.to_numpy()
yDataTest = yDataTest.to_numpy()

xDataTrain = xDataTrain.astype(np.float32)
yDataTrain = yDataTrain.astype(np.float32)
xDataTest = xDataTest.astype(np.float32)
yDataTest = yDataTest.astype(np.float32)

# Save data as pickle
with open('yDataTrain.pkl', 'wb') as f:
            pickle.dump(yDataTrain, f)

with open('xDataTrain.pkl', 'wb') as f:
            pickle.dump(xDataTrain, f)

with open('yDataTest.pkl', 'wb') as f:
            pickle.dump(yDataTest, f)

with open('xDataTest.pkl', 'wb') as f:
            pickle.dump(xDataTest, f)
