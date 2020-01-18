import matplotlib.pyplot as plt
import cv2
img = cv2.imread('test.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off') # axis 출력 끄기
plt.show()
