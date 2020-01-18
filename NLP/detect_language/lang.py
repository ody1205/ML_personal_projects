'''
언어 판정을 할때 언어에 따라 다른 문자를 사용하기 때문에 한글로 쓰여있으면 한국어 히라가나와 한자로 쓰여 있으면 일본어 등 이지만,
영어, 프랑스어, 스페인어 등은 모두 알파벳으로 쓰여있음. 이러한 경우 문자의 빈도를 사용해 언어를 판정.

LinearSVC 알고리즘으로 분류할 수 없는 경우나 텍스트 데이터의 경우 나이브 베이즈(Naive Bayes)를 사용.
3가지 종류의 나이브 베이즈 학습기가 있지만 그 중 가장 간단한 GaussianNB를 사용해봄.
'''
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def count_codePoint(str):
    counter = np.zeros(65535) # Unicode basic plane 0, 0 ~ 65535(FFFF)까지 저장할 배열 준비
    
    for i in range(len(str)):
        code_point = ord(str[i])
        if code_point > 65535:
            continue
        counter[code_point] += 1
    
    counter = counter / len(str)
    return counter

ko_str = '이것은 한국어 문장입니다.'
en_str = 'This is English sentences.'

x_train = [count_codePoint(ko_str), count_codePoint(en_str)]
y_train = ['ko', 'en']

clf = GaussianNB()
clf.fit(x_train, y_train)

ko_test_str = '안녕하세요'
en_test_str = 'Hello'

x_test = [count_codePoint(ko_test_str), count_codePoint(en_test_str)]
y_test = ['ko', 'en']

y_pred = clf.predict(x_test)
print(y_pred)
print(accuracy_score(y_test, y_pred))
