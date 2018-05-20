from keras.models import Sequential
from keras.layers import Dense
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("data/pima-indians-diabetes.data.csv", delimiter=",")
# split into input (X) and output (Y) variables
datasetX = dataset[:,0:8]
datasetY = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callback for tensorboard
import keras
tbCallBack = keras.callbacks.TensorBoard(log_dir='/tmp/keras_logs', write_graph=True)

# Fit the model
model.fit(datasetX, datasetY, epochs=150, batch_size=10,  verbose=1, validation_split=0.3, callbacks=[tbCallBack])

# evaluate the model
scores = model.evaluate(datasetX, datasetY)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))