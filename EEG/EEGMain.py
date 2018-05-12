import BuildEvents
import ReadEDF
import numpy as np 
import os

BaseDirEval = '/Volumes/KINGSTON/eval'
BaseDirTrain = '/Volumes/KINGSTON/train'
fs = 250
TrainFeatures = np.zeros([1, fs])
TrainLabels = np.zeros([1,1])
EvalFeatures = np.zeros([1, fs])
EvalLabels = np.zeros([1,1])

for dirName, subdirList, fileList in os.walk(BaseDirEval):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname[-4:] == '.edf':
            print('\t%s' % fname)
            [sig, event] = ReadEDF.readEDF(dirName + '/' + fname)
            [l, f] = BuildEvents.BuildEvents(sig, event)
            TrainFeatures = np.append(TrainFeatures, f, axis = 0)
            TrainLabels = np.append(TrainLabels, l, axis = 0)
"""
for dirName, subdirList, fileList in os.walk(BaseDirTrain):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname[-4:] == '.edf':
            print('\t%s' % fname)
            [sig, event] = ReadEDF.readEDF(dirName + '/' + fname)
            [l, f] = BuildEvents.BuildEvents(sig, event)
            TrainFeatures = np.append(TrainFeatures, f, axis = 0)
            TrainLabels = np.append(TrainLabels, l, axis = 0)
"""
np.savetxt('EEGFeatures.txt', TrainFeatures, delimiter=",")
np.savetxt('EEGLabels.txt', TrainLabels, delimiter=",")
print(TrainFeatures.shape)
print(TrainLabels.shape)

