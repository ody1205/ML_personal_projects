import matplotlib.pyplot as plt
import cv2
from mosaic import mosaic as mosaic

img = cv2.imread('./test_imgs/cat.jpg')
mos = mosaic(img, (50,50,450,450), 10)

cv2.imwrite('cat-mosaic.png', mos)
plt.imshow(cv2.cvtColor(mos, cv2.COLOR_BGR2RGB))
plt.show()