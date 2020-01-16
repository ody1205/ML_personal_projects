import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('tem10y.csv', encoding='utf-8')

md = {}

# # 일 평균 구하기
# for i, row in data.iterrows():
#     m, d, v = (int(row['월']), int(row['일']), int(row['기온']))
#     key = str(m) + '/' + str(d)
#     if key not in md:
#         md[key] = []
#     md[key] += [v]

# avs = {}
# for key in md:
#     v = avs[key] = sum(md[key]) / len(md[key])
#     print('{0} : {1}'.format(key, v))

# 일 평균 쉽게 구하기
f = data.groupby(['월','일'])['기온']
ff = f.sum()/f.count()
# print(ff)

# 월 평균 구하기
g = data.groupby(['월'])['기온']
gg = g.sum()/g.count()

# print(gg)
# gg.plot()
# plt.show()

#기온이 30도 이상 넘는 날 구하기
hot_bool = (data['기온'] > 30)
hot = data[hot_bool]

cnt = hot.groupby(['연'])['연'].count()

# print(cnt)

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_year = data['연'] <= 2015
test_year = data['연'] >= 2016
interval = 6

def make_data(data):
    x = []
    y = []
    temps = list(data['기온'])
    for i in range(len(temps)):
        if i < interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
        x.append(xa)
    return (x,y)
train_x, train_y = make_data(data[train_year])
test_x, test_y = make_data(data[test_year])

lr = LinearRegression(normalize = True)
lr.fit(train_x, train_y)
pred_y = lr.predict(test_x)

# plt.figure(figsize=(10,6), dpi=100)
# plt.plot(test_y, c='r')
# plt.plot(pred_y, c='b')
# plt.show()

diff_y = abs(pred_y - test_y)
print('average = ', sum(diff_y)/len(diff_y))
print('max = ', max(diff_y))