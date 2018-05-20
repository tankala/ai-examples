import pandas as pds

dataframeX = pds.read_csv('data/KaggleV2-May-2016.csv', usecols= [2, 5, 7, 8, 9, 10, 11, 12])
dataframeY = pds.read_csv('data/KaggleV2-May-2016.csv', usecols=[13])

def genderToInt(gender):
    if gender == 'M':
        return 0
    else:
        return 1

def statusToInt(status):
    if status == 'No':
        return 0
    else:
        return 1

dataframeX.Gender = dataframeX.Gender.apply(genderToInt)
dataframeY.Noshow = dataframeY.Noshow.apply(statusToInt)

print(dataframeX.head())
print(dataframeY.head())

# 1
import numpy as np
seed = 7
np.random.seed(seed)

# 2
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(9, input_shape=(8,), init='uniform', activation='sigmoid'))
model.add(Dense(9, init='uniform', activation='sigmoid'))
model.add(Dense(9, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.summary()

# 3
import keras
tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

# 4
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(dataframeX.values, dataframeY.values, epochs=9, batch_size=50,  verbose=1, validation_split=0.3, callbacks=[tbCallBack])