import matplotlib.pyplot as plt
import numpy as np


def x2(a, npl, b):
    return [a*n*n+10*b for n in npl]


x = np.linspace(-2,2)

plt.ylim(-10,10)

plt.plot(x, x2(8, x, 0.75), color='r')
plt.plot(x, x2(6, x, 0.5), color='g')
plt.plot(x, x2(4, x, 0.25), color='b')
plt.plot(x, x2(2, x, 0), color='k')
plt.text(5, 5, "Rețeaua \nGenerator")
plt.grid(True, which='both')

plt.plot(x, x2(-0.5, x, 0), color='k')
plt.plot(x, x2(-1, x, -0.25), color='b')
plt.plot(x, x2(-1.5, x, -0.5), color='g')
plt.plot(x, x2(-2, x, -0.75), color='r')
plt.text(5, -5, "Rețeaua \nDiscriminator")

plt.plot(x, [2.5] * len(x), 'g-')
plt.plot(x, [-2.5] * len(x), 'g-')

plt.show()
