import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math 

def distance_color(f, ax, x, y):
    distances = np.abs(f) 
    norm = plt.Normalize(distances.min(), distances.max())
    colors = cm.plasma(norm(distances))
    for i in range(len(f) - 1):
        ax.plot(x[i:i + 2], y[i:i + 2], color=colors[i], linewidth=2)


def fig_1(t, s):
    fig, axs = plt.subplots(2, figsize=(16, 9))
    infasurare = [s[i] * math.e ** (-2 * np.pi * 1j *  i / len(s)) for i in range(len(s))]
    
    distance_color(s, axs[0], t, s)
    distance_color(infasurare, axs[1], np.real(infasurare), np.imag(infasurare))

    axs[0].stem(t[380], s[380], linefmt='k-', markerfmt='ko', basefmt=' ')
    axs[1].plot([0, np.real(infasurare[380])], [0, np.imag(infasurare[380])], 'k-', linewidth=2)
    axs[1].plot(np.real(infasurare[380]), np.imag(infasurare[380]), 'ko', markersize=6)

    axs[0].stem(t[420], s[420], linefmt='r-', markerfmt='ro', basefmt=' ')
    axs[1].plot([0, np.real(infasurare[420])], [0, np.imag(infasurare[420])], 'r-', linewidth=2)
    axs[1].plot(np.real(infasurare[420]), np.imag(infasurare[420]), 'ro', markersize=6)    

    axs[0].set_xlabel('amplitudine')
    axs[0].set_ylabel('timp')
    axs[1].set_xlabel('real')
    axs[1].set_ylabel('imaginar')

    axs[1].set_box_aspect(1)
    axs[0].grid(True)
    axs[1].grid(True)
    plt.tight_layout()
    plt.savefig('../plots/2_fig1.png', format='png', dpi=300) 
    plt.savefig('../plots/2_fig1.pdf', format='pdf')

def fig_2(s, omega):
    fig, axs = plt.subplots(2, 2, figsize = (16, 9)) 


    for o, ax in enumerate(axs.flat):
        infasurare = [s[i] * math.e ** (-2 * np.pi * 1j * omega[o] * i / len(s)) for i in range(len(s))]
        
        distance_color(infasurare, ax, np.real(infasurare), np.imag(infasurare))
        ax.set_xlabel('real')
        ax.set_ylabel('imaginar')
        ax.grid(True)
        ax.set_title(r'$\omega = {}$'.format(omega[o]))
        ax.set_box_aspect(1)
    
    plt.suptitle('Infasurarea unui semnal sinusoidal cu frecventa = 15')
    plt.tight_layout()
    plt.savefig('../plots/2_fig2.png', format='png', dpi=300) 
    plt.savefig('../plots/2_fig2.pdf', format='pdf')

t = np.linspace(0, 1, 1000)
f = 15 
s = np.sin(2 * np.pi * t * f)
omega = [3, 7, 12, 15]

fig_1(t, s)
fig_2(s, omega)
