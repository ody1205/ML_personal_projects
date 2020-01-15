from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 학습 전용 데이터와 결과값 준비
train_data = [[0,0],[1,0],[0,1],[1,1]] # 학습 전용 데이터 X and Y
train_label = [0,1,1,0] # X xor Y 실제 결과값

# SVM. LinearSVC 알고리즘 지정하기
clf = LinearSVC()

# 학습 전용 데이터와 결과 학습하기
clf.fit(train_data, train_label)

# 테스트 데이터로 결과 예측하기
test_data = [[0,0],[1,0],[0,1],[1,1]]
test_label = clf.predict(test_data)

print(test_data, '의 예측결과: ', test_label)
print(accuracy_score([0,1,1,0], test_label))
# 결과가 좋지 못함. 25% 예측 결과값을 도출해냄. 다른 알고리즘 사용필요.

# LinearSVC -> NOT WORKING -> TEXT DATA -> NO -> KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier

# KNN 알고리즘 지정하기
knn = KNeighborsClassifier(n_neighbors = 1)

# 학습 전용 데이터와 결과 학습하기
knn.fit(train_data, train_label)

# 테스트 데이터로 결과 예측하기
knn_test_label = knn.predict(test_data)

print(test_data, '의 예측결과: ', knn_test_label)
print(accuracy_score([0,1,1,0], knn_test_label))

# 100% 예측 결과값을 도출해냄.
