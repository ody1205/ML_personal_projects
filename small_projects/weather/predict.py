from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('export_dataframe.csv', encoding = 'utf-8')
train_year = (df['연'] <= 2017) # 2010.01.01 ~ 2017.12.31
test_year = (df['연'] >= 2018) # 2018.01.01 ~ 2019.01.01
interval = 6

def make_data(data):
    x = []
    y = []
    temps = list(data['온도'])
    for i in range(len(temps)):
        if i < interval:
            continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x,y)

train_x, train_y = make_data(df[train_year])
test_x, test_y = make_data(df[test_year])

lr = LinearRegression(normalize=True)
lr.fit(train_x, train_y)
pre_y = lr.predict(test_x)

diff_y = abs(pre_y - test_y)
print('accuracy of the model against train data = ',lr.score(train_x, train_y)) # measuring accuracy of the model against the train data
print('classificationreport= ',lr.score(test_x, test_y)) # equivalent to classificationreport(test_y, pre_y)
print('평균 오차값=', sum(diff_y)/len(diff_y))
print('최대 오차값=', max(diff_y))

plt.figure(figsize=(10,6), dpi=100)
plt.plot(test_y, c='r')
plt.plot(pre_y, c='b')
plt.savefig('weather-tem-lr.png')
plt.show()

