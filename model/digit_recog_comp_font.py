import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.utils import np_utils

#Path to dataset images
path = "C:\\Users\Rohan\Desktop\datasets\Fnt\\"
os.chdir(path)
folder_path = path + "Sample0"
folder_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

#Creating dataset examples
dataset = []
dataset_label = []
i = -1
for name in folder_names:
    directory = folder_path + name
    os.chdir(directory)
    i += 1
    for filename in os.listdir(directory):
        image = cv2.imread(filename, 0)
        h, w = image.shape
        r = 28 / w
        dim = (28, int(r * h))
        image = cv2.resize(image, dim)
        image = image.reshape((1, 784)).astype('float32')
        dataset.append(image)
        dataset_label.append(i)

dataset = np.array(dataset)
dataset_label = np.array(dataset_label)

#print(dataset.shape)
#print(dataset_label.shape)

dataset = dataset.reshape((dataset.shape[0], 784)).astype('float32')

'''
#Plot an image from dataset
i = 3
image = dataset[i]
image = image.reshape((28,28)).astype('float32')
plt.imshow(image, cmap='Greys')
plt.show()
print(dataset_label[i])
'''

#shuffling and dividing the dataset examples into training and testing set
indices = np.random.permutation(dataset.shape[0])
np.take(dataset, indices, axis=0, out=dataset)
np.take(dataset_label, indices, axis=0, out=dataset_label)

b = int(dataset.shape[0] * 0.8)
xtrain = dataset[:b]
ytrain = dataset_label[:b]
xtest = dataset[b:]
ytest = dataset_label[b:]

#print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

num_pixels = 784
num_classes = 10

xtrain = xtrain / 255
xtest = xtest / 255

ytrain = np_utils.to_categorical(ytrain)
ytest = np_utils.to_categorical(ytest)

l0 = tf.keras.layers.Dense(num_pixels, input_dim=num_pixels, activation='relu', kernel_initializer='normal')
l1 = tf.keras.layers.Dense(num_classes, activation='softmax', kernel_initializer='normal')
model = tf.keras.Sequential([l0, l1])

history = model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=10, batch_size=10, verbose=2)

scores = model.evaluate(xtest, ytest, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model.save("C://Users/Rohan/Desktop/model/digit_recog_comp_font.h5")

'''
#Predicting from test dataset
i = 89
image = xtest[i].reshape((28,28)).astype('float32')
#print(image)
plt.imshow(image, cmap='Greys')
plt.show()
image = xtest[i].reshape((1, 784)).astype('float32')
prediction = model.predict(image)
print(prediction.argmax())
'''

'''
#Predicting from custom images
import cv2
image = cv2.imread("C:\\Users\Rohan\Desktop\digits\\image0.jpg", 0)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

h, w = image.shape[:2]
r = 28/w
dim = (28, int(r*h))
image = cv2.resize(image, dim)

for i in range(28):
    for j in range(28):
        if image[i][j] < 150:
            image[i][j] = 255
        else:
            image[i][j] = 0          

cv2.imshow("", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image2 = image.reshape((1, 784)).astype('float32')
image2 = image2 / 255
#print(image2)
prediction = model.predict(image2)
print(prediction.argmax())
'''