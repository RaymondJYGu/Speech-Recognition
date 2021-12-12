
# Import statements

import os
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import warnings
import time
import sounddevice as sd
import soundfile as sf
from keras.models import load_model
from keras.models import Model

warnings.filterwarnings("ignore")



# Load the model
model= load_model('best_model.hdf5')

# Defining the classifications
classes= ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']


def predict(audio):
    prob=model.predict(audio.reshape(1,8000,1))
    index=np.argmax(prob[0])
    return classes[index]



# Parameters for input sound data

samplerate = 16000
duration = 1  # seconds
filename = 'fun.wav'


# Make prediction based on user's input
while(True):
    print("start")
    mydata = sd.rec(int(samplerate * duration), samplerate=samplerate,
        channels=1, blocking=True)
    print("end")
    sd.wait()
    sf.write(filename, mydata, samplerate)
    samples, sample_rate = librosa.load('fun.wav', sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    print(predict(samples))
    print(3)
    time.sleep(1)
    print(2)
    time.sleep(1)
    print(1)
    time.sleep(1)
