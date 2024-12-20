import numpy as np
import matplotlib.pyplot as plt
import requests

fig, axs = plt.subplots(2, 2, figsize=(16, 9))

response = requests.get("https://qrng.anu.edu.au/API/jsonI.php?length=100&type=uint8")

t = np.arange(100)
x = response.json()['data'] 
n = 0
for i in range(2):
    for j in range(2):
        if(n != 0):
            x = np.convolve(x, x, mode = 'same')
        axs[i][j].plot(t, x)
        axs[i][j].grid()
        axs[i][j].set_title(f'Convolutia numarul {n}')
        n += 1

plt.tight_layout()
plt.savefig('../plots/1.png', dpi = 200)
plt.savefig('../plots/1.pdf')
