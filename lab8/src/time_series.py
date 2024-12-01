import matplotlib.pyplot as plt
import numpy as np

def save_plot(time, signal, title, filename):
    plt.figure(figsize=(16, 9))
    plt.plot(time, signal)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig('../plots/' + filename + '.pdf')
    plt.clf()


def autocorrelation(signal):
    autocorrelation = []
    N = len(signal)
    for k in range(-N + 1, N): 
        r_k = sum(signal[i] * signal[i + abs(k)] for i in range(N - abs(k))) / (N - abs(k))  
        autocorrelation.append(r_k)
    return autocorrelation 

def ar_model(signal, m, p):
    Y = np.array([signal[-p-i:len(signal)-i] for i in range(m)])
    y = signal[-m:]
    
    x = np.linalg.inv(Y.T @ Y) @ Y.T @ y

    return x
   

# a)
polinom = {'a': 3, 'b': 5, 'c': 4}
freq = [12, 18]
time = np.linspace(0, 1, 1000)

trend = polinom['a'] * time ** 2 + polinom['b'] * time + polinom['c']
seasonal = np.sin(2 * np.pi * freq[0] * time) + np.sin(2 * np.pi * freq[1] * time)
white_noise = np.random.normal(0, 0.3, 1000)

signal = trend + seasonal + white_noise

save_plot(time, trend, 'Trend', 'trend')
save_plot(time, seasonal, 'Seasonal', 'seasonal')
save_plot(time, white_noise, 'White Noise', 'white_noise')
save_plot(time, signal, 'Final Signal', 'final_signal')


# b)
t = np.linspace(0, 1, 1999)
save_plot(t, autocorrelation(signal), 'Autocorrelation', 'autocorrelation')

auto_corr = np.correlate(signal, signal, mode='full')
save_plot(t, auto_corr, 'Numpy Autocorrelation', 'np_autocorrelation')


# c) 
p = 100 
m = 150

predictions = signal.copy()
for k in range (500):
    x_star = ar_model(predictions, m, p)
    prediction = x_star.T @ predictions[-p:]
    
    predictions = np.append(predictions, prediction)

time = np.linspace(0, 1.5, 1500)
plt.figure(figsize=(16, 9))
plt.plot(time[:1000], predictions[:1000], label="Signal", color="purple")  
plt.plot(time[-500:], predictions[-500:], label="Predictions", color="gold")
plt.title(f"AR Model predictions on 500 consecutive samples\nm = {m}, p = {p}")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig('../plots/c_ar_model.pdf')
plt.clf()


# d)
