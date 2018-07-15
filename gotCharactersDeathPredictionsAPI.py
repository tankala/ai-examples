import flask
import numpy as np
import tensorflow as tf
from keras.models import load_model

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
def init():
    global model,graph
    # load the pre-trained Keras model
    model = load_model('models/gotCharactersDeathPredictions.h5')
    graph = tf.get_default_graph()

# API for prediction
@app.route("/predict", methods=["GET"])
def predict():
    nameOfTheCharacter = flask.request.args.get('name')
    parameters = getParameters()
    inputFeature = np.asarray(parameters).reshape(1, 12)
    with graph.as_default():
        raw_prediction = model.predict(inputFeature)[0][0]
    if raw_prediction > 0.5:
        prediction = 'Alive'
    else:
        prediction = 'Dead'
    return sendResponse({nameOfTheCharacter: prediction})

# Getting Parameters
def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('male'))
    parameters.append(flask.request.args.get('book1'))
    parameters.append(flask.request.args.get('book2'))
    parameters.append(flask.request.args.get('book3'))
    parameters.append(flask.request.args.get('book4'))
    parameters.append(flask.request.args.get('book5'))
    parameters.append(flask.request.args.get('isMarried'))
    parameters.append(flask.request.args.get('isNoble'))
    parameters.append(flask.request.args.get('numDeadRelations'))
    parameters.append(flask.request.args.get('boolDeadRelations'))
    parameters.append(flask.request.args.get('isPopular'))
    parameters.append(flask.request.args.get('popularity'))
    return parameters

# Cross origin support
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    init()
    app.run(threaded=True)