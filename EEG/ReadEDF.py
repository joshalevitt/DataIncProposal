import mne
import numpy as np

def readEDF(fileName):
    Rawdata = mne.io.read_raw_edf(fileName)
    signals = Rawdata.get_data()
    RecFile = fileName[0:-3] + 'rec'
    eventData = np.genfromtxt(RecFile, delimiter = ",")
    Rawdata.close()
    return [signals, eventData]


