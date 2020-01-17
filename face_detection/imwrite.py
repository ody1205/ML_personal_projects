import cv2

img = cv2.imread('test.png')

cv2.imwrite('out.png', img)