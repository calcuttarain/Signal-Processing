import matplotlib.pyplot as plt
import numpy as np

def rectangular_window(nw):
    return [1] * nw

def hanning_window(nw):
    return [0.5 * (1 - np.cos((2 * np.pi * i) / (nw - 1))) for i in range (nw)]

def hamming_window(nw):
    return [0.54 - 0.46 * np.cos((2 * np.pi * i) / (nw - 1)) for i in range(nw)]

def blackman_window(nw):
    return [0.42 - 0.5 * np.cos((2 * np.pi * i) / (nw - 1)) + 0.08 * np.cos((4 * np.pi * i) / (nw - 1)) for i in range(nw)]

def flat_top_window(nw):
    return [0.22 - 0.42 * np.cos((2 * np.pi * i) / (nw - 1)) + 0.28 * np.cos((4 * np.pi * i) / (nw - 1)) - 0.08 * np.cos((6 * np.pi * i) / (nw - 1)) + 0.007 * np.cos((8 * np.pi * i) / (nw - 1)) for i in range(nw)]


f = 100
nw = 200

t = np.linspace(0, 1, 1600)
s = np.sin(2 * np.pi * f * t)

r_window = np.resize(rectangular_window(nw), 1600)
hn_window = np.resize(hanning_window(nw), 1600)

fig, axs = plt.subplots(2, figsize=(16, 9))

axs[0].plot(t, s * r_window)
axs[0].grid()
axs[0].set_title('Rectangular Window')

axs[1].plot(t, s * hn_window)
axs[1].grid()
axs[1].set_title('Hanning Window')

plt.tight_layout()
plt.savefig('../plots/3.png', dpi = 200)
plt.savefig('../plots/3.pdf')
plt.clf()


fig, axs = plt.subplots(3, 2, figsize = (16, 9))
axs[2, 1].axis('off')

windows = {'Rectangular Window': rectangular_window, 'Hanning Window': hanning_window, 'Hamming Window': hamming_window, 'Blackman Window': blackman_window, 'Flat Top Window': flat_top_window}

for ax, (name, window_func) in zip(axs.flat, windows.items()):
    window = np.array(window_func(nw))
    padded_window = np.concatenate((window, np.zeros(len(t) - nw)))  
    
    ax.plot(t, s * padded_window)
    ax.set_title(name)
    ax.grid(True)
    ax.set_xlim([0, 0.3])
    ax.set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig('../plots/3_windows.png', dpi=200)
plt.savefig('../plots/3_windows.pdf')
