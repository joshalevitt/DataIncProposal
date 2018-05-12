from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import BuildEvents
import ReadEDF
import os

tf.logging.set_verbosity(tf.logging.INFO)

#This network was adapted from https://www.tensorflow.org/tutorials/layers

# Our application logic will be added here
def cnn_model(features, labels, mode):
    inputLayer = tf.reshape(features["x"], [-1, 250, 1])

    conv1 = tf.layers.conv1d(
        inputs = inputLayer,
        filters = 40,
        kernel_size = [10],
        padding = "same",
        activation = tf.nn.relu
    )

    pool1 = tf.layers.max_pooling1d(
        inputs = conv1,
        pool_size=[2],
        strides = 2
    )

    conv2 = tf.layers.conv1d(
        inputs = pool1,
        filters = 40,
        kernel_size = [10],
        padding = "same",
        activation = tf.nn.relu
    )

    pool2 = tf.layers.max_pooling1d(
        inputs = conv2,
        pool_size = [5],
        strides = 5
    )

    pool2Flat = tf.reshape(pool2, [-1, 25 * 40])

    dense = tf.layers.dense(
        inputs = pool2Flat,
        units = 600,
        activation = tf.nn.relu
    )
    dropout = tf.layers.dropout(
        inputs = dense,
        rate = 0.4,
        training = mode == tf.estimator.ModeKeys.TRAIN
    )
    logits = tf.layers.dense(
        inputs = dropout,
        units = 6,
        #use_bias = False
    )

    predictions = {
        "classes": tf.argmax(input = logits, axis = 1),
        "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(learning_rate = 0.005, momentum = 0.1)
        train_op = optimizer.minimize(
            loss = loss,
            global_step = tf.train.get_global_step()
        )
        
        return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels = labels, predictions = predictions["classes"]),
        "ROC": tf.metrics.auc(labels = labels, predictions = predictions["classes"])
    }

    return tf.estimator.EstimatorSpec(
        mode = mode,
        loss = loss,
        eval_metric_ops = eval_metric_ops
    )


# Load Data
def main(argv):


    BaseDirEval = '/Volumes/KINGSTON/eval'
    BaseDirTrain = '/Volumes/KINGSTON/train'
    fs = 250
    features = np.zeros([1, fs])
    labels = np.zeros([1,1])
    

    for dirName, subdirList, fileList in os.walk(BaseDirEval):
        print('Found directory: %s' % dirName)
        for fname in fileList:
            if fname[-4:] == '.edf':
                print('\t%s' % fname)
                [sig, event] = ReadEDF.readEDF(dirName + '/' + fname)
                [l, f] = BuildEvents.BuildEvents(sig, event)
                features = np.append(features, f, axis = 0)
                labels = np.append(labels, l, axis = 0)

    [numSamp, q] = features.shape
    r = np.random.RandomState(1000)
    perm = r.permutation(numSamp)
    features = features[perm]
    labels = labels[perm]
    #features = np.genfromtxt('/Users/joshualevitt/Desktop/DataInc/EEG/EEGFeatures.txt', delimiter=',')
    TrainFeatures = np.asarray(features[1:200000, :], dtype = np.float32)
    TestFeatures = np.asarray(features[200001:-1, :], dtype = np.float32)
    print('Features loaded...')
    #labels = np.genfromtxt('/Users/joshualevitt/Desktop/DataInc/EEG/EEGLabels.txt', delimiter=',')
    TrainLabels = np.asarray(labels[1:200000], dtype = np.int32)
    TestLabels = np.asarray(labels[200001:-1],  dtype = np.int32)
    print('Labels loaded...')

    EEG_Classifier = tf.estimator.Estimator(
        model_fn = cnn_model,
        model_dir = "/tmp/eeg_convnet_model_mo2"
    )
    print('CNN struct loaded...')

    """
    predict_inputs_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": TestFeatures},
        num_epochs = 1,
        shuffle =False
    )

    pred = EEG_Classifier.predict(
        input_fn = predict_inputs_fn,
    )

    pred = list(pred)
    confMat = np.zeros([6,6])
    numPred = len(pred)
    for i in range(numPred): 
        prediction = pred[i]["classes"]
        actual = TestLabels[i]
        confMat[prediction, actual] += 1
    print(confMat)
    """

    tensors_to_log = {"probablilities" : "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": TrainFeatures},
        y = TrainLabels,
        batch_size = 100,
        num_epochs = None,
        shuffle = True
    )

    print('Beginning Training...')

    EEG_Classifier.train(
        input_fn = train_input_fn,
        steps = 200,
        hooks = [logging_hook]
    )

    print('Beginning Testing...')
    eval_inputs_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": TestFeatures},
        y = TestLabels,
        num_epochs = 1,
        shuffle = False
    )

    eval_results = EEG_Classifier.evaluate(input_fn = eval_inputs_fn)
    print(eval_results)

    predict_inputs_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": TestFeatures},
        num_epochs = 1,
        shuffle =False
    )

    pred = EEG_Classifier.predict(
        input_fn = predict_inputs_fn,
    )

    pred = list(pred)
    confMat = np.zeros([6,6])
    numPred = len(pred)
    for i in range(numPred): 
        prediction = pred[i]["classes"]
        actual = TestLabels[i]
        confMat[prediction, actual] += 1
    print(confMat)

   
if __name__ == "__main__":
  tf.app.run()