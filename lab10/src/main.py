import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg

def save_plot(time, signal, title, filename):
    plt.figure(figsize=(16, 9))
    plt.plot(time, signal)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    # plt.savefig('../plots/' + filename + '.pdf')
    plt.show()
    plt.clf()

# 1
polinom = {'a': 3, 'b': 5, 'c': 4}
freq = [12, 18]
time = np.linspace(0, 1, 1000)

trend = polinom['a'] * time ** 2 + polinom['b'] * time + polinom['c']
seasonal = np.sin(2 * np.pi * freq[0] * time) + np.sin(2 * np.pi * freq[1] * time)
white_noise = np.random.normal(0, 0.3, 1000)

signal = trend + seasonal + white_noise

# 2
p = 20  
model = AutoReg(signal, lags=p)
model_fitted = model.fit()

reconstructed_signal = model_fitted.fittedvalues

plt.figure(figsize=(16, 9))
plt.plot(time, signal, label="Original Signal")
plt.plot(time[p:], reconstructed_signal, label="Reconstructed Signal", linestyle='--')
plt.title("Original vs Reconstructed Signal using AR model")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
