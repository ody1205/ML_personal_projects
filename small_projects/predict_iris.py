import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# csv 데이터 불러오기
data = pd.read_csv('iris.csv', encoding='utf-8')

# y값에 레이블 저장하기
y = data.loc[:,'Name']
# x값에 데이터 저장하기
x = data.loc[:,['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]

# x, y 값을 트레이닝(80%)과 테스트 데이터(20%)로 나누기. 셔플기능을 활성화해 데이터에 랜덤계수 추가.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# SVC 알고리즘 불러오기
clf = SVC()

# SVC 알고리즘으로 데이터 트레이닝 시키기
clf.fit(x_train, y_train)

# 테스트 데이터로 예측값 도출하기
y_pred = clf.predict(x_test)

# 정확도 측정
print('정확도 = ',accuracy_score(y_test, y_pred))