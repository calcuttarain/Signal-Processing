import numpy as np
import matplotlib.pyplot as plt

def a():
    f_0 = 400
    t = np.linspace(0, 0.3, 1600)

    y = np.sin(2 * np.pi * t * f_0)

    plt.figure()
    plt.plot(t, y)
    plt.savefig("../plots/2_a.pdf")

def b():
    f_0 = 800
    t = np.linspace(0, 3, 10000)

    y = np.sin(2 * np.pi * t * f_0)

    plt.figure()
    plt.plot(t, y)
    plt.savefig("../plots/2_b.pdf")

def c():
    f_0 = 240
    t = np.linspace(0, 0.2, 1600)
    y = t * f_0 % 1

    plt.figure()
    plt.plot(t, y)
    plt.savefig("../plots/2_c.pdf")

def d():
    f_0 = 300
    t = np.linspace(0, 0.2, 1600)
    y = np.sign(np.sin(2 * np.pi * t * f_0))

    plt.figure()
    plt.plot(t, y)
    plt.savefig("../plots/2_d.pdf")

def e():
    i = np.random.rand(128, 128)
    
    plt.figure()
    plt.imshow(i)
    plt.savefig("../plots/2_e.pdf")

def f():
    arrays = [np.ones(128) if (i // 16 % 2) else np.zeros(128) for i in range (128)]
    i = np.vstack(arrays)

    plt.figure()
    plt.imshow(i)
    plt.savefig("../plots/2_f.pdf")

a()

b()

c()

d()

e()

f()
