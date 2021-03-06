import matplotlib.pyplot as plt
import numpy as np
from math import exp

e = exp(1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.ylabel('Vc(t)')
plt.xlabel('t')

t = np.linspace(0, 10)

r = 2 * e**(-t/2) / 3**(1/2) * np.sin(3**(1/2) * t / 2)

vc = 1 - e**(-t/2) * (3**(1/2)/3 * np.sin(3**(1/2)/2 * t) + np.cos(3**(1/2)/2 * t))


plt.annotate('t = 3.62759, Vc = 1.16303353481537', xy=(3.62759, 1.16303353481537), xytext=(2.5, 1.25),
             arrowprops=dict(facecolor='black', shrink=0.05),)

plt.plot(t, vc)

y = 1 - e**(-3.62759/2) * (3**(1/2)/3 * np.sin(3**(1/2)/2 * 3.62759) + np.cos(3**(1/2)/2 * 3.62759))
print(y)

plt.show()
