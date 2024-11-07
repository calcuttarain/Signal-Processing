import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# a) frecventa de esantionare e de un esantion pe ora => 1Hz


# b) 18288 ore = 762 zile


# c) frecventa maxima = frecventa de esantionare / 2 = 1 / 2 = 0.5


# d)
data = pd.read_csv('../data/Train.csv')
raw_s = data.iloc[:, 2].astype(float).values

fs = 1  
n = len(raw_s)
dc_offset = np.mean(raw_s)

s = raw_s - dc_offset

complex_ft = np.fft.fft(s)
ft = complex_ft[:n // 2]
ft = np.abs(ft / n)

f = fs * np.linspace(0, n / 2, n // 2) / n

plt.stem(f, ft)
plt.ylabel("|X(ω)|")
plt.xlabel("frecventa")
plt.tight_layout()
plt.grid()
plt.savefig('../plots/d.png')
plt.savefig('../plots/d.pdf')
plt.clf()


# e) semnalul are o componenta continua deoarece transformata Fourier are modulul foarte mare la valoarea de 0 Hz => 
# scadem media semnalului
raw_ft = np.fft.fft(raw_s)
raw_ft = raw_ft[:n // 2]
raw_ft = np.abs(raw_ft / n)

plt.plot(f, raw_ft)
plt.title('transformata semnal cu componenta continua')
plt.xlim(0, 0.002)
plt.grid()
plt.tight_layout()
plt.savefig('../plots/e_1.png')
plt.savefig('../plots/e_1.pdf')
plt.clf()

plt.plot(f, ft)
plt.title('transformata semnal fara componenta continua')
plt.xlim(0, 0.002)
plt.grid()
plt.tight_layout()
plt.savefig('../plots/e_2.png')
plt.savefig('../plots/e_2.pdf')
plt.clf()


# f)
sorted_indices = np.argsort(ft)[::-1] 
sorted_ft = ft[sorted_indices]
sorted_f = f[sorted_indices]

top_4_ft = sorted_ft[:4]
top_4_f = sorted_f[:4]

top_4_f_ore = [1 / freq for freq in top_4_f]

print(top_4_ft) # [66.85385766 35.21917298 27.10202229 25.21991648] -> cele mai mari module ale transformatei
print(top_4_f) # [5.46866455e-05 1.09373291e-04 4.16712239e-02 1.64059937e-04] -> frecventele lor (in ore^(-1))
print(top_4_f_ore) # [18286.0, 9143.0, 23.997375328083987, 6095.333333333333] -> frecventele lor (in ore)

# 18286 ore ~ 762 zile
# 9143 ~ 380 zile ~ 1 an: pot indica sarbatorile de iarna  
# 23.99: indica un comportament similar de-a lungul zilei, probabil legat de faptul ca circula mai putine masini noaptea decat ziua
# 6095.33 ~ 253 zile ~ 8 luni: probabil vacantele scolare de vara au un impact


# g)
luna = s[1560:2304] + dc_offset
timp_luna = np.arange(0, len(luna))

zile_etichete = np.arange(0, len(luna), 24)  
zile_saptamana = [str((zi % 7) + 1) for zi in range(len(zile_etichete))] 

plt.plot(timp_luna, luna)
plt.xticks(ticks=zile_etichete, labels=zile_saptamana)
plt.title('Plotarea semnalului pe o luna cu etichetele pe zile ale saptamanii')
plt.xlabel("zile ale săptămânii")
plt.ylabel("numar de masini")
plt.tight_layout()
plt.grid()
plt.savefig('../plots/g.pdf')
plt.savefig('../plots/g.png')
plt.clf()

# h)
'''
In prima instanta, am putea studia comportamentul semnalului la frecvente de 24 de ore pentru a-l putea localiza in ore ale zilei. De exemplu, traficul mai scazut noaptea si mai ridicat in anumite ore de varf ale zilei.
Apoi putem deduce din frecventele saptamanale zilele saptamanii. La fel, traficul mai scazut in weekend si mai aglomerat in timpul saptamanii.
In mod similar, se poate proceda cu frecvente pe perioade mai extinse, cum ar fi cateva luni pentru a putea localiza semnalul intr-un an. Speram, astfel, sa existe un anumit tipar pentru sarbatori / anotimpuri.
In ultima instanta, putem incerca sa observam cresterea numarului de masini pentru a-l putea plasa efectiv intre anumiti ani. 

Probabil ca cea mai mare limitare pe care o are aceasta metoda e ca nu se cunoaste exact locul de unde a fost extras semnalul. Asta nu afecteaza atat de mult comportamentul zilnic sau saptamanal, dar devine o problema cand incercam sa facem deduceri pe perioade mai mari care depind de anotimpuri/sarbatori specifice unor anumite zone. 
'''

# i) nu ma intereseaza un comportament al semnalului pe o perioada mai scurta decat cea zilnica => 
# consider zgomot frecventele mai inalte decat cea pe 24 de ore gasita la subpunctul g) si le elimin
t = np.arange(0, len(s))

plt.plot(t, s)
plt.title('Semnalul nefiltrat')
plt.xlabel('timp(ore)')
plt.ylabel('numar masini')
plt.grid()
plt.tight_layout()
plt.savefig('../plots/semnal_nefiltrat.pdf')
plt.savefig('../plots/semnal_nefiltrat.png')
plt.clf()

# index = np.where(f == top_4_f[2])[0][0]
# ft_filtered = np.copy(ft)
# ft_filtered[index:] = 0
#
# ft_full = np.concatenate((ft_filtered, np.conj(ft_filtered[::-1]))) / n
# s_filtered = np.fft.ifft(ft_full).real
# plt.plot(t, s_filtered)
# plt.title('Semnalul filtrat')
# plt.xlabel('timp(ore)')
# plt.ylabel('numar masini')
# plt.grid()
# plt.tight_layout()
# plt.show()
# plt.savefig('../plots/semnal_nefiltrat.pdf')
# plt.savefig('../plots/semnal_nefiltrat.png')
#plt.clf()
