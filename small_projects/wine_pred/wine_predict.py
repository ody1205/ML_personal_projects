'''
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 데이터 불러오기
data = pd.read_csv('winequality-white.csv', sep=';', encoding='utf-8')

# y값에 레이블 저장하기
y = data['quality']
# x값에 데이터 저장하기
x = data.drop('quality', axis = 1)

# x,y 테스트 트레인 데이터로 나누기
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# 랜덤포레스트 알고리즘 불러오기
rf = RandomForestClassifier()

# 랜덤포레스트 알고리즘 사용해 트레이닝 하기
rf.fit(x_train, y_train)

# 예측값 y_pred에 저장하기
y_pred = rf.predict(x_test)

print(classification_report(y_test,y_pred))
print('Accuracy = ', accuracy_score(y_test, y_pred))

# 정답률이 63% ~ 67% 사이로 굉장히 낮음. UndefinedMetricWarning을 고쳐봐야겠음.

'''

# import matplotlib.pyplot as plt

# # 데이터를 그룹해 갯수 계산
# count_data = data.groupby('quality')['quality'].count()
# print(count_data)

# count_data.plot()
# plt.savefig('wine-count-plt.png')
# plt.show()

'''
그래프를 보면 와인의 품질 데이터의 대부분이 5~7이고 이 외에는 거의 없는걸 볼 수 있습니다.
따라서 등급을 4이하, 5~7 사이, 8이상이라는 3개로 분류합니다.
'''
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 데이터 불러오기
data = pd.read_csv('winequality-white.csv', sep=';', encoding='utf-8')

# y값에 레이블 저장하기
y = data['quality']
# x값에 데이터 저장하기
x = data.drop('quality', axis = 1)

new_list = []
for l in list(y):
    if l <= 4:
        new_list += [0]
    elif l <= 7:
        new_list += [1]
    else:
        new_list += [2]

y = new_list

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
'''
정답률이 93% 대로 상승한것을 볼수있습니다.
새로 만든 와인 레이블의 정의를 내리면
0 = 품질이 나쁜 와인
1 = 보통의 와인
2 = 품질이 좋은 와인
'''