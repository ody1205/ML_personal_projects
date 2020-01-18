import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import glob

'''
counting unicode appearance in a given string and return counter.
'''
def count(str):
    counter = np.zeros(65535) # create an empty list to hold Unicode values from 0000 (0) to FFFF (65535)

    for i in range(len(str)):
        uni = ord(str[i]) # change each characters to unicode
        if uni > 65535: # if we see a character that is not within plane 0 unicode, skip
            continue
        counter[uni] += 1 # increment counter for the character

    counter = counter/len(str)
    return counter

index = 0
x_train = []
y_train = []
# train set preperation
for file in glob.glob('./train/*.txt'):
    y_train.append(file[8:10]) # since ./train/ takes up 7 spaces start from 8 and the length of language indicator is 2 ends at 10

    file_str = ''
    for line in open(file, 'r'):
        file_str = file_str + line # append every lines in the file to file_str
    x_train.append(count(file_str)) # send all the lines to the count method

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