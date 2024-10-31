import scipy.io.wavfile
import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math 
import time

def read_samples():
    files = ['a.wav', 'e.wav', 'i.wav', 'o.wav', 'u.wav']
    samples = []
    rate = 0
    for file in files:
        rate, data = scipy.io.wavfile.read('../sounds/' + file)
        samples.append(data)
    return rate, samples

def group_samples(samples):
    groups = []
    for sample in samples:
        group = np.split(sample, 200)
        groups.append([np.concatenate([group[i], group[i + 1]]) for i in range (len(group) - 2)])
    return groups

def compute_dft(groups, rate):
    dftm = []
    for group in groups:
        lines = []

        for signal in group:
            dft = np.fft.fft(signal)
            magnitude = np.abs(dft)
            lines.append(magnitude)
        dftm.append(np.column_stack(lines))
        return dftm

rate, samples = read_samples()

groups = group_samples(samples)

dftm = compute_dft(groups, rate)
