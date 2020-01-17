import matplotlib.pyplot as plt
import cv2
from mosaic import mosaic as mosaic

# cascade_file = 'haarcascade_frontalface_alt.xml'
# cascade = cv2.CascadeClassifier('./haarcascade/' + cascade_file)

# img = cv2.imread('./test_imgs/family.jpg')
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# face_list = cascade.detectMultiScale(img_gray, minSize= (150,150))
# for (x,y,w,h) in face_list:
#     img = mosaic(img, (x,y,x+w, y+h), 10)

# cv2.imwrite('family-mosaic.png', img)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
'''
앞 모습 모자이크 = haarcascade_frontalface_alt.xml
옆 모습 모자이크 = haarcascade_profileface.xml (이건 왼쪽 얼굴만으로 training 되어있기 때문에 오른쪽 얼굴을 찾을때 이미지를 반전 시켜야함.)
'''

cascade_file = 'haarcascade_profileface.xml'
cascade = cv2.CascadeClassifier('./haarcascade/' + cascade_file)

side_img = cv2.imread('./test_imgs/yoko.jpg')
img_gray = cv2.cvtColor(side_img, cv2.COLOR_BGR2GRAY)

flipped = cv2.flip(img_gray, 1)
face_list = cascade.detectMultiScale(flipped, minSize = (150,150))

for (x,y,w,h) in face_list:
    side_img = mosaic(side_img, (x, y, x+w, y+h), 10)
cv2.imwrite('side_mosaic.png', side_img)
plt.imshow(cv2.cvtColor(side_img, cv2.COLOR_BGR2RGB))
plt.show()
