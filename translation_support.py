'''Import necessary libraries'''
'''Make the necessary imports'''
import os
import numpy as np
from numpy import mean
import random
from math import ceil, floor
import json
from scipy.signal import hamming 
import pandas as pd
import numpy as np
import sys
import parselmouth
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import medfilt
from vosk import Model, KaldiRecognizer
import sys
import pickle

model = Model("model")

low_idx = int(sys.argv[1])
high_idx = int(sys.argv[2])

rec = KaldiRecognizer(model, 16000)

for idx in range(low_idx, high_idx+1):
    file_location = 'temp_dir/' + str(idx) + '.pkl'

    if str(idx) + '.pkl' in os.listdir('temp_dir/'):
        with open(file_location, 'rb') as f:
            data = pickle.load(f)

        transcription = ""
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            transcription += " " + res['text']

        words = transcription.split(" ")

        with open(file_location.replace(".pkl", ".json"), 'w') as f:
            json.dump(words, f)
