# load GOT characters death dataset
import pandas as pds
dataframeX = pds.read_csv('data/character-predictions.csv', usecols= [7, 16, 17, 18, 19, 20, 25, 26, 28, 29, 30, 31])
dataframeY = pds.read_csv('data/character-predictions.csv', usecols=[32])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataframeX.values, dataframeY.values, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fix random seed for reproducibility
import numpy as np
seed = 7
np.random.seed(seed)

# create model
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(15, input_dim=12, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# log for tensorboard graph purpose
import keras
tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

# Fit the model
model.fit(X_train, Y_train, epochs=100, batch_size=50,  verbose=1, callbacks=[tbCallBack])

# Save the model
model.save('models/gotCharactersDeathPredictions.h5')

# Predicting the Test set results
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test, Y_pred)
print("\nConfusion Matrix:")
print(cm)
acs = accuracy_score(Y_test, Y_pred)
print("\nAccuracy Score: %.2f%%" % (acs * 100))
