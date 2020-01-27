import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('export_dataframe.csv')
g = df.groupby(['월'])['온도']
gg = g.sum() / g.count()
print(gg)
gg.plot()
plt.savefig('temp-mont-avg.png')
plt.show()