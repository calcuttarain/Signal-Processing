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

ft = np.fft.fft(s)
ft = ft[:n // 2]
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

top_4_f_ore = []
for i in top_4_f:
    top_4_f_ore.append(1/i)

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
