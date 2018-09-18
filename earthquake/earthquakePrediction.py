import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras



quakeDataset = pd.read_csv('../../Datasets/database.csv')
quakeDataset = quakeDataset.reindex(np.random.permutation(quakeDataset.index))
# print(quakeDataset.head())
# print(" {} {}" .format( "info before parsing date" , quakeDataset.info()))
quakeDataset['parsed_date'] = pd.to_datetime(quakeDataset['Date'] , format="%m/%d/%Y" , infer_datetime_format= True)
input_features = quakeDataset[[ "Latitude", "Longitude" ]]
target_feature = quakeDataset[["Magnitude"]]
test_input  = input_features.head(12000)
test_target = target_feature.head(12000)
validation_input = input_features.tail(11412)
validation_target =target_feature.tail(11412)

print("test label details", test_target.info())
print("description of test target" , test_target.describe(), "\n")
print("validation target details", validation_target.info())
print("description of validation target" , validation_target.describe(), "\n")

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation= 'relu'))
model.add(keras.layers.Dense(64, activation= 'relu'))
model.add(keras.layers.Dense(units = 1, input_dim = 10, activation='softmax'))
model.compile(optimizer = tf.train.AdamOptimizer() , loss='mse' , metrics=['mae'])
model.fit(test_input.values,test_target.values,epochs = 10,batch_size = 50,validation_data = (validation_input.values,validation_target.values))








