import tensorflow as tf
import numpy as np
import BuildEvents
import ReadEDF
import os
import pickle

n_classes = 6
batch_size = 1000
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

x = tf.placeholder(tf.float32, [None, 250])
y = tf. placeholder(tf.float32, [None])
fs = 250
"""
BaseDirEval = '/Volumes/KINGSTON/eval'
BaseDirTrain = '/Volumes/KINGSTON/train'

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
labels = labels.astype(int)
tempLabels = np.zeros([numSamp, n_classes])
tempLabels[np.arange(numSamp), labels] = 1
labels = tempLabels

with open('EEGfeatures.pickle', 'wb') as f:
        pickle.dump(features, f)
with open('EEGlabels.pickle', 'wb') as f:
        pickle.dump(labels, f)   

"""
with open('EEGfeatures.pickle', 'rb') as f:
        features = pickle.load(f)
with open('EEGlabels.pickle', 'rb') as f:
        labels = pickle.load(f)
[numSamp, q] = features.shape
#"""

r = np.random.RandomState(1000)
perm = r.permutation(numSamp)
features = features[perm]
labels = labels[perm]
partition = 320000
#features = np.genfromtxt('/Users/joshualevitt/Desktop/DataInc/EEG/EEGFeatures.txt', delimiter=',')
TrainFeatures = np.asarray(features[1:partition, :], dtype = np.float32)
TestFeatures = np.asarray(features[partition:-1, :], dtype = np.float32)
print('Features loaded...')
#labels = np.genfromtxt('/Users/joshualevitt/Desktop/DataInc/EEG/EEGLabels.txt', delimiter=',')
TrainLabels = np.asarray(labels[1:partition, :], dtype = np.int32)
TestLabels = np.asarray(labels[partition:-1, :],  dtype = np.int32)
print('Labels loaded...')
[numTrain, _] = TrainLabels.shape



def getBatch(n):
    batch = np.zeros([1,fs])
    batchLabels = np.zeros([1,n_classes]) 
    ind = np.random.randint(0, numTrain, n)
    for j in ind:
        batch = np.append(batch, np.reshape(TrainFeatures[j, :],(1, 250)), axis=0)
        batchLabels = np.append(batchLabels, np.reshape(TrainLabels[j, :], (1, 6)), axis=0)
    batch = np.delete(batch, 0, 0)
    batchLabels = np.delete(batchLabels, 0, 0)
    return {'Labels': batchLabels, 'Features': batch}



def conv1d(x, W):
    return tf.nn.conv1d(x, W, stride = 1, padding = 'SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 1, 1], strides = [1, 2, 1, 1], padding = 'SAME')

def convolutional_neural_network(x): #, keep_rate):
    weights = {
        # 5 x 5 convoltion, 1 input image, 32 ouputs
        'W_conv1': tf.Variable(tf.random_normal([10, 1, 32])),
        # 5 x 5 conv, 32 inputs, 64 outputs
        'W_conv2': tf.Variable(tf.random_normal([10, 32, 64])),
        # fully connected, 7*7*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([63*64, 1024])),
        # 1024 inputs, 10 outputs (class prediciton)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Reshape input to a 4D tensor
    x = tf.reshape(x, shape = [-1, 250, 1])
    # Convolution Layer, using out function
    conv1 = tf.nn.relu(conv1d(x, weights['W_conv1']) + biases['b_conv1'])
    # Max Pooling (down-sampling)
    conv1 = tf.reshape(conv1, shape = [-1, 250, 32, 1])
    conv1 = maxpool2d(conv1)
    conv1 = tf. reshape(conv1, shape = [-1, 125, 32])
    # Convolution Layer
    conv2 = tf.nn.relu(conv1d(conv1, weights['W_conv2']) + biases['b_conv2'])
    # max pooling
    conv2 = tf.reshape(conv2, shape = [-1, 125, 64, 1])
    conv2 = maxpool2d(conv2)

    #fully connected Layer
    # reshape conv2 output to fit fully connected layer
    fc = tf.reshape(conv2, [-1, 63*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels = y)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for b in range(int(numTrain/batch_size)):
                DataDict = getBatch(batch_size)
                epoch_x = DataDict['Features']
                epoch_y = DataDict['Labels']
                epoch_y = [ np.where(r==1)[0][0] for r in epoch_y ].astype(int)
                
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:', accuracy.eval({x: TestFeatures, y: TestLabels}))

                _, c = sess.run([optimizer, cost], feed_dict = {x: epoch_x, y: epoch_y})
                print ('batch', b, 'out of', int(numTrain/batch_size), 'in epoch', epoch, 'loss:', c)
                epoch_loss += c

            print ('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: TestFeatures, y: TestLabels}))

train_neural_network(x)