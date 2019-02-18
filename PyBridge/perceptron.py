import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

f = open('data/results/linear_separator_model.txt', mode='r')
buf = f.read()
print(buf)
p = re.compile(r'(-?\d+(\.?\d+)?)')
print(p.findall(buf))
weights = [float(x[0]) for x in p.findall(buf)]
print(weights)

pd_data = pd.read_csv('./data/train/linearly_separable.txt', sep=", ", engine="python")
class1 = pd_data.loc[pd_data['y'] == 1][['x1', 'x2']]
class2 = pd_data.loc[pd_data['y'] == -1][['x1', 'x2']]

print(class2)
x = np.linspace(-6, 10, 500)
y = -weights[0]/weights[2] - weights[1]/weights[2]*x
plt.plot(class1['x1'].values.tolist(), class1['x2'].values.tolist(), 'ro')
plt.plot(class2['x1'].values.tolist(), class2['x2'].values.tolist(), 'go')
plt.plot(x, y)
plt.ylim(-4, 6)
plt.show()
