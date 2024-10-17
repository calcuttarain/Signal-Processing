import numpy as np
import scipy.io.wavfile
import scipy.signal
import sounddevice

def a(t):
    f_0 = 400
    y = np.sin(2 * np.pi * t * f_0)
    return y

def b(t):
    f_0 = 800
    y = np.sin(2 * np.pi * t * f_0)
    return y

def c(t):
    f_0 = 240
    y = t * f_0 % 1
    return y

def d(t):
    f_0 = 300
    y = np.sign(np.sin(2 * np.pi * t * f_0))
    return y

t = np.linspace(0, 3, 3 * 44100)
sounds = [a(t), b(t), c(t), d(t)]

for sound in sounds:
    sounddevice.play(sound, 44100)
    sounddevice.wait()

rate = int(10e5)
scipy.io.wavfile.write('../sounds/3.wav', rate, sounds[0])
rate, x = scipy.io.wavfile.read('../sounds/3.wav')
sounddevice.play(x, 44100)
sounddevice.wait()
