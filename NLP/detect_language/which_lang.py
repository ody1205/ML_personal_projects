import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import glob

def count(str):
    counter = np.zeros(65535)

    for i in range(len(str)):
        uni = ord(str[i])
        if uni > 65535:
            continue
        counter[uni] += 1

    counter = counter/len(str)
    return counter

index = 0
x_train = []
y_train = []

for file in glob.glob('./train/*.txt'):
    y_train.append(file[8:10])

    file_str = ''
    for line in open(file, 'r'):
        file_str = file_str + line
    x_train.append(count(file_str))

clf = GaussianNB()
clf.fit(x_train, y_train)

index = 0
x_test = []
y_test = []
for file in glob.glob('./test/*.txt'):
    y_test.append(file[7:9])

    file_str = ''
    for line in open(file, 'r'):
        file_str = file_str + line
    x_test.append(count(file_str))

y_pred = clf.predict(x_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))