from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

train_data = [[0,0], [0,1], [1,0], [1,1]]
train_label = [0, 1, 1, 1]

clf = LinearSVC()
clf.fit(train_data, train_label)

test_data = [[0,0], [0,1], [1,0], [1,1]]
test_label = clf.predict(test_data)

print(test_data, '데이터의 예측결과: ',test_label)
print(accuracy_score([0,1,1,1], test_label))