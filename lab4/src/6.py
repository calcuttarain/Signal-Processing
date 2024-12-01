import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

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

def plot_spectrograms(dftm, rate):
    vocals = ['a', 'e', 'i', 'o', 'u']

    for i, sound in enumerate(dftm):
        frequencies = np.fft.fftfreq(sound.shape[0], 1 / rate)
        times = np.arange(sound.shape[1]) 
        
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        sound = sound[positive_freq_idx, :]

        plt.figure(figsize=(16, 9))
        plt.pcolormesh(times, frequencies, 10 * np.log10(sound), shading='gouraud', cmap='inferno')
        plt.title(f'Vocala {vocals[i]}')
        plt.ylabel('Frecventa')
        plt.xlabel('Esantioane')
        plt.colorbar(label='Intensitate')
        plt.tight_layout()

        plt.savefig('../plots/6_' + vocals[i] + '.pdf')
        plt.clf()

def plot_full_spectrogram(dftm, rate):
    combined_dftm = np.hstack(dftm)

    frequencies = np.fft.fftfreq(combined_dftm.shape[0], 1 / rate)

    positive_freq_idx = frequencies >= 0
    frequencies = frequencies[positive_freq_idx]
    combined_dftm = combined_dftm[positive_freq_idx, :]

    plt.figure(figsize=(16, 9))
    plt.pcolormesh(np.arange(combined_dftm.shape[1]), frequencies, 10 * np.log10(combined_dftm), shading='gouraud', cmap='inferno')
    plt.title('Spectograma')
    plt.ylabel('Frecventa')
    plt.xlabel('Esantioane')
    plt.colorbar(label='Intensitate')
    plt.tight_layout()

    plt.savefig('../plots/6_' + 'spectograma' + '.pdf')
    plt.clf()

rate, samples = read_samples()

groups = group_samples(samples)

dftm = compute_dft(groups, rate)

plot_spectrograms(dftm, rate)

plot_full_spectrogram(dftm, rate)
