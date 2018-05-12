import ReadEDF
import numpy as np

def BuildEvents(signals, EventData):
    [numEvents, z] = (EventData.shape)
    fs = 250.0
    samplesPerEvent = 11
    [numChan, numPoints] = signals.shape
    for i in range(numChan):
        if np.std(signals[i, :]) > 0:
            signals[i, :] = (signals[i, :] - np.mean(signals[i, :])) / np.std(signals[i, :])
    features = np.zeros([numEvents * samplesPerEvent, int(fs)])
    labels = np.zeros([numEvents * samplesPerEvent, 1])
    for i in range(numEvents):
        chan = int(EventData[i, 0])
        start = round(EventData[i, 2] * fs)
        [numChan, numPoints] = signals.shape
        for sample in range(samplesPerEvent):
            startPoint = int(round(start + (sample - samplesPerEvent/2) * (fs / samplesPerEvent)))
            endPoint = int(startPoint + fs)
            if endPoint <= numPoints:
                features[i * samplesPerEvent + sample, :] = signals[chan, startPoint:endPoint]
                labels[i * samplesPerEvent + sample, 0] = int(EventData[i, 3]) - 1
        
    return [labels, features]




