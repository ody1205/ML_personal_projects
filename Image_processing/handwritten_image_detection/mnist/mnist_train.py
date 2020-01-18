from mnist import MNIST
import random

mndata = MNIST('./sample_data/')

images, labels = mndata.load_training()

# images, labels = mndata.load_testing()
for i in range(10):
    print(mndata.display(images[i]))