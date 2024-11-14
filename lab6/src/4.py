import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

data = pd.read_csv('../data/Train.csv')
raw_s = data.iloc[:, 2].astype(float).values

ticks = np.arange(0, 73, 6)
labels = [(f"{int(hour%24):02}:00") for hour in ticks]


# a)
t = np.arange(24 * 3)
s = raw_s[5664:5736]

plt.figure(figsize=(16, 9))
plt.plot(t, s)
plt.xticks(ticks, labels)
plt.grid()
plt.xlabel('ore')
plt.ylabel('numar masini')
plt.title('Semnal Nefiltrat')
plt.tight_layout()
plt.savefig('../plots/semnal_nefiltrat.png', dpi = 200)
plt.savefig('../plots/semnal_nefiltrat.pdf')
plt.clf()


# b)
nw = [5, 9, 13, 17]
fig, ax = plt.subplots(figsize=(16, 9))

ax.plot(t, s, label = f'semnal original')

for w in nw:
    filtered_s = np.convolve(s, np.ones(w), 'valid') / w
    tw = np.arange(0, 71, 71 / len(filtered_s))
    ax.plot(tw, filtered_s, label = f'w = {w}')

ax.legend().set_title('Dimensiunea ferestrei')
ax.grid()
ax.set_xlabel('ore')
ax.set_ylabel('masini')
plt.title('Filtru de tip medie alunecatoare')
plt.savefig('../plots/4_b.png', dpi = 200)
plt.savefig('../plots/4_b.pdf')
plt.clf()


# c) 
'''
Rata de esantionare este de un esantion pe ora => consideram ca semnalul are o frecventa maxima de 1/2, iar orice altceva e zgomot (conform criteriului de esantionare Nyquist).
In ore, consideram zgomot frecvente mai mari de 2 ore.
In Hz, semnalul de frecventa maxima e 1/3600 (frecventa de esantionare) * 1/2 = 1/7200
'''
wn = 0.5 


# d)
b_butter, a_butter = scipy.signal.butter(5, wn, btype = 'low')
b_cheby, a_cheby = scipy.signal.cheby1(5, 1, wn, btype = 'low')


# e)
s_butter = scipy.signal.filtfilt(b_butter, a_butter, s)
s_cheby = scipy.signal.filtfilt(b_cheby, a_cheby, s)

plt.figure(figsize=(16, 9))
plt.plot(t, s, label = 'Semnal Nefiltrat')
plt.plot(t, s_butter, label = 'Semnal Filtrat Butterworth')
plt.plot(t, s_cheby, label = 'Semnal Filtrat Chebyshev')
plt.title('Filtrare Butterworth vs Filtrare Chebyshev')
plt.xticks(ticks, labels)
plt.legend()
plt.grid()
plt.savefig('../plots/4_d.png', dpi = 200)
plt.savefig('../plots/4_d.pdf')
plt.clf()
'''
Filtrul Chebyshev pare sa reflecte mai bine semnalul, pe cand Butterworth pare sa-l netezeasca prea mult.
'''


# f)
N = [2, 3, 5, 8, 9]
RP = [1, 3, 5, 7, 10]

plt.figure(figsize=(16, 9))
plt.plot(t, s, label = 'semnal nefiltrat')
for n in N:
    b_butter, a_butter = scipy.signal.butter(n, wn, btype = 'low')
    s_butter = scipy.signal.filtfilt(b_butter, a_butter, s)
    plt.plot(t, s_butter, label = f'N = {n}')

plt.title('Filtrare Butterworth')
plt.legend().set_title('Ordinul filtrului')
plt.xticks(ticks, labels)
plt.grid()
plt.savefig('../plots/4_f_butterworth_ord.png', dpi = 200)
plt.savefig('../plots/4_f_butterworth_ord.pdf')
plt.clf()
'''
Schimbarea ordinului in cazul filtrului Butterworth nu are o importanta semnificativa in forma semnalului filtrat.
'''


plt.figure(figsize=(16, 9))
plt.plot(t, s, label = 'semnal nefiltrat')
for n in N:
    b_cheby, a_cheby = scipy.signal.cheby1(n, 5, wn, btype = 'low')
    s_cheby = scipy.signal.filtfilt(b_cheby, a_cheby, s)
    plt.plot(t, s_cheby, label = f'N = {n}')

plt.title('Filtrare Chebyshev')
plt.legend().set_title('Ordinul filtrului')
plt.xticks(ticks, labels)
plt.grid()
plt.savefig('../plots/4_f_chebyshev_ord.png', dpi = 200)
plt.savefig('../plots/4_f_chebyshev_ord.pdf')
plt.clf()
'''
La Chebyshev, in cazul unui ordin par se schimba drastic amplitudinea. 
Filtrul de ordin 9 compromite prea multa informatie din semnalul original.
Filtrele de ordin 3 si 5 ofera un echilibru bun intre eliminarea zgomotului si forma semnalului.
'''


plt.figure(figsize=(16, 9))
plt.plot(t, s, label = 'semnal nefiltrat')
for rp in RP:
    b_cheby, a_cheby = scipy.signal.cheby1(5, rp, wn, btype = 'low')
    s_cheby = scipy.signal.filtfilt(b_cheby, a_cheby, s)
    plt.plot(t, s_cheby, label = f'rp = {rp}')

plt.title('Filtrare Chebyshev')
plt.legend().set_title('RP (dB)')
plt.xticks(ticks, labels)
plt.grid()
plt.savefig('../plots/4_f_chebyshev_rp.png', dpi = 200)
plt.savefig('../plots/4_f_chebyshev_rp.pdf')
plt.clf()
'''
Pentru rp = 1, se pastreaza prea mult din zgomot.
Pentru rp = 7, 9, se pierde prea multa informatie.
Echilibrul ramane in continuare la valorile 3, 5.
'''


ordin = 3
rp = 5

b_final, a_final = scipy.signal.cheby1(ordin, rp, wn, btype = 'low')
s_filtrat = scipy.signal.filtfilt(b_final, a_final, s)

plt.figure(figsize=(16, 9))
plt.plot(t, s_filtrat)
plt.title('Semnal Filtrat')
plt.xticks(ticks, labels)
plt.grid()
plt.savefig('../plots/semnal_filtrat.png', dpi = 200)
plt.savefig('../plots/semnal_filtrat.pdf')
