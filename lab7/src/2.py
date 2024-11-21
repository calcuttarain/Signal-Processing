from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

'''
SNRdB = 10 * log10(PowerSignal / PowerNoise)

PowerS = the mean square of the signal values
'''

def computeSNR(original_signal, filtered_signal):
    p_signal = np.mean(original_signal ** 2)
    p_noise = np.mean((original_signal - filtered_signal) ** 2)

    return 10 * np.log10(p_signal / p_noise)


def filter_signal(signal, cutoff, order = 5):
    nyquist = 0.5 * signal.shape[1]  
    wn = cutoff / nyquist  
    
    b, a = butter(order, wn, btype='low')  
    
    filtered_signal = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        filtered_signal[i, :] = filtfilt(b, a, signal[i, :])  

    for i in range(signal.shape[1]):
        filtered_signal[:, i] = filtfilt(b, a, filtered_signal[:, i])

    return filtered_signal
    

X = misc.face(gray=True)
SNR = [70, 80, 90, 100]

print(np.mean(X**2))
for snr in SNR:
    compressed_image_snr = 0 
    cutoff = 0.2

    while True:
        compressed_image = filter_signal(X, cutoff)
        compressed_image_snr = computeSNR(X, compressed_image)
        
        if abs(compressed_image_snr - snr) < 0.1:
            break 
        elif compressed_image_snr > snr:
            cutoff -= 0.1
        else:
            cutoff += 0.1

        print(abs(compressed_image_snr - snr))


