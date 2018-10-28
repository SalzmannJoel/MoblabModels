import numpy as np
import tensorflow as tf
import tensorflow.contrib.lite as tflite
from keras.datasets import mnist
from keras.utils import np_utils
np.random.seed(1671) # for reproducibility
# network and training
NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number of digits
OPTIMIZER = tf.keras.optimizers.SGD() # SGD optimizer, explained later in this chapter
N_HIDDEN = 128
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
# data: shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
#X_train is 60000 rows of 28x28 values --> reshaped in 60000 x 784
RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(Y_test, NB_CLASSES)


def createModel():
    # 10 outputs
    # final stage is softmax
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(NB_CLASSES, input_shape=(RESHAPED,)))
    model.add(tf.keras.layers.Activation('softmax'))
    model.summary()
    model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=OPTIMIZER, metrics=[tf.keras.metrics.categorical_accuracy])
    return model


def train(model, x_train, y_train):
    # train
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE, validation_split=VALIDATION_SPLIT)
    return model


def evaluate(model):
    score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
    print("Test score:", score[0])
    print('Test accuracy:', score[1])


# This method takes a model, loads the weights from the weightsPath-File and predicts the given value
def predict(model, weightsPath, value):
    # Predict a given Array
    model.load_weights(weightsPath)
    probabilities = model.predict(np.expand_dims(value, axis=0))
    printAnswer(probabilities)


# This method creates a new model, trains it on X_train as zeros and ones, Y_train and predicts the given value
def predictMNISTasZeroOrOne(value):
    model = createModel()
    newSet = getMNISTasZeroOrOne()
    trainedModel = train(model, newSet, Y_train)
    probabilities = trainedModel.predict(np.expand_dims(value, axis=0))
    printAnswer(probabilities)


# This method creates a new model, trains it on X_train, Y_train and predicts the given value
def predictMNIST(value):
    model = createModel()
    trainedModel = train(model, X_train, Y_train)
    probabilities = trainedModel.predict(np.expand_dims(value, axis=0))
    printAnswer(probabilities)


#  This method creates a new model, trains it on X_train, Y_train and predicts the given value
def printAnswer(probabilities):
    answer = 0
    answerProbability = 0
    for index, item in enumerate(probabilities[0]):
        if item > answerProbability:
            answer = index
    print("I think it is a " + str(answer))


def saveModel(model, filePath="keras_model.h5"):
    # Save tf.keras model in HDF5 format.
    keras_file = filePath
    tf.keras.models.save_model(model, keras_file)


def convert(saveFileName, kerasFile="keras_model.h5"):
    # Convert to TensorFlow Lite model.
    # converter = tflite.convert_savedmodel.convert(keras_file, "converted_model.tflite")
    converter = tflite.TFLiteConverter.from_keras_model_file(kerasFile)
    tflite_model = converter.convert()
    open(saveFileName+".tflite", "wb").write(tflite_model)


def getMNISTasZeroOrOne():
    newTrainSet = X_train
    for sampelIndex, vector in enumerate(newTrainSet):
        print("\n" + str(sampelIndex))
        for index, value in enumerate(vector):
            if value != 0:
                vector[index] = 1
    return newTrainSet


model = createModel()
newSet = getMNISTasZeroOrOne()
trainedModel = train(model, newSet, Y_train)
saveModel(trainedModel, "modelFromMNISTZeroesAndOnes.h5")
convert("modelFromMNISTZeroesAndOnes", "modelFromMNISTZeroesAndOnes.h5")
