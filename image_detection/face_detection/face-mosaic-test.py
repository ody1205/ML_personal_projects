import matplotlib.pyplot as plt
from mosaic import mosaic as mosaic
import cv2

cascade_files = ['haarcascade_frontalface_alt.xml', 'haarcascade_profileface.xml']

img = cv2.imread("./test_imgs/test2.jpeg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

flipped = cv2.flip(img_gray, 1)
for i in cascade_files:
    cascade = cv2.CascadeClassifier('./haarcascade/' + i)
    face_list = cascade.detectMultiScale(img_gray, minSize= (150,150))
    if len(face_list) == 0:
        flipped = cv2.flip(img_gray, 1)
        face_list = cascade.detectMultiScale(flipped, minSize= (150,150))
    for (x,y,w,h) in face_list:
        img = mosaic(img, (x, y, x+w, y+h), 10)

cv2.imwrite('mosaic-img.png', img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

