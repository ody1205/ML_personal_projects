import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()
'''
데이터셋에 손글씨 숫자들 확인하기
subplot을 사용해 앞 15개 숫자들 확인
'''
# for i in range(15):
#     plt.subplot(3, 5, i+1)
#     plt.axis('off')
#     plt.title(str(digits.target[i]))
#     plt.imshow(digits.images[i], cmap='gray')
# plt.show()

'''
숫자가 어떻게 픽셀로 변환되어 출력되는지 확인하기
0 은 배경을 나타내며 1 ~ 16 사이의 숫자들로 흰색 선을 나타냄
'''
# d0 = digits.images[0]
# plt.imshow(d0, cmap='gray')
# plt.show()
# print(d0)

'''
이미지
'''