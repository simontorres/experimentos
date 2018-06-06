import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re

import pandas as pd


df = pd.DataFrame.from_csv('all_peugeot.csv')
print(df.columns.values)
# df.plot(y='Precio')
# df.plot.show()
df = df.sort_values(by='Anho')
# kms


precio = [int(re.sub('[$.]','', df['Precio'][i])) for i in range(len(df['Precio']))]
anho = [int(y) for y in df['Anho']]
for k in df['Kilometraje']:
    print(k, type(k), pd.isna(k))
kilometraje = [int(re.sub('[kms.]','', k))  if not pd.isna(k) else 0 for k in df['Kilometraje']]
print(kilometraje)

scale = [200 * (k / max(kilometraje)) for k in kilometraje]
colors = [cm.jet(color) for color in scale]
print(colors)

plt.scatter(anho, precio, marker='o', s=scale, cmap='jet', alpha=0.5)

# plt.scatter(anho, kilometraje, marker='*', color='r')
# plt.ylim((min(precio), max(precio)))
plt.show()