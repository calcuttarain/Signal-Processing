import numpy as np
import sounddevice as sd
import scipy.signal
import scipy.io.wavfile

def signal(f):
    t = np.linspace(0, 3, 44100 * 3)
    return np.sin(2 * np.pi * f * t)

f1 = 400
f2 = 800 

combined_signal = np.concatenate((signal(f1), signal(f2)))

sd.play(combined_signal, 44100) 
sd.wait()

scipy.io.wavfile.write('../sounds/5.wav', int(10e5), combined_signal)

#cu cat semnalul are frecventa mai mare, cu atat sunetul este mai inalt
