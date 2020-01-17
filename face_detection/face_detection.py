import matplotlib.pyplot as plt
import cv2

cascade_file = "haarcascade_profileface.xml"
cascade = cv2.CascadeClassifier('./haarcascade/'+cascade_file)

img = cv2.imread("./test_imgs/yoko.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

flipped = cv2.flip(img_gray, 1)
face_list = cascade.detectMultiScale(flipped, 1.3, 5)

# face_list = cascade.detectMultiScale(img_gray, minSize=(150, 150))

if len(face_list) == 0:
    print("실패")
    quit()

for (x,y,w,h) in face_list:
    print("얼굴의 좌표 =", x, y, w, h)
    red = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x+w, y+h), red, thickness=5)


cv2.imwrite("face-detect.png", img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()