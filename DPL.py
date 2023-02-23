import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
from tensorflow.python.keras import datasets, layers, models

from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

(train_images, train_labels), (test_image, test_labels) = datasets.cifar10.load_data()

# normalize data bw 0 and 1
train_images, test_images = train_images / 255.0, test_image / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Image generator to expand training dataset to reduce model bias toward training data and provide variety
datagen = ImageDataGenerator(
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')


n = train_images.shape[0]

# create 4 new images from existing training dataset by using data augmentation
for i in range(n):
    img = image.img_to_array(train_images[i])  # convert image to numpy arry
    img = img.reshape((1,) + img.shape)
    j = 0
    for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):
        # append new augmented data to training set with the correct shape
        train_images = np.append(train_images, np.reshape(batch[0], [1, 32, 32, 3]), axis = 0) 
        train_labels = np.append(train_labels, np.reshape(train_labels[i], [1, 1]), axis = 0)
        j += 1
        if j > 3:  # stop after generating 4 variations of the each training image
           break

# make model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# Model compilation
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=4, 
                    validation_data=(test_images, test_labels))


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)