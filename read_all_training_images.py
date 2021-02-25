from MNistBinaryManager import MNistBinaryManager
import numpy as np
import matplotlib.pyplot as plt

# imports for tensorflow stuff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# constants

training_label_path = './training_sets_mnist/train-labels.idx1-ubyte'
training_image_path = './training_sets_mnist/train-images.idx3-ubyte'
test_label_path = './training_sets_mnist/t10k-labels.idx1-ubyte'
test_image_path = './training_sets_mnist/t10k-images.idx3-ubyte'

# load in the binaries into numpy arrays
mnb = MNistBinaryManager(training_label_path, training_image_path, test_label_path, test_image_path)

'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(mnb.getBinaryByTag(training_image_path)[i] / 255, cmap=plt.cm.binary)
    plt.xlabel(mnb.getBinaryByTag(training_label_path)[i])
plt.show()
'''

# gets the shape keras will by using the training image lengths
keras_shape = (len(mnb.getBinaryByTag(training_image_path)[0]), len(mnb.getBinaryByTag(training_image_path)[1]))

model = keras.Sequential([
  keras.layers.Flatten(input_shape=keras_shape),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10)
])

model.compile(
optimizer='adam',
loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
metrics=['accuracy']
)

model.fit(mnb.getBinaryByTag(training_image_path)/255, mnb.getBinaryByTag(training_label_path), epochs=10)

test_loss, test_acc = model.evaluate(mnb.getBinaryByTag(test_image_path)/255, mnb.getBinaryByTag(test_label_path), verbose=2)
print("\nTest accuracy:", test_acc)

probability_model = keras.Sequential(
[model, keras.layers.Softmax()]
)

predictions = probability_model.predict(mnb.getBinaryByTag(test_image_path)/255)

#print(predictions[0])

#print(np.argmax(predictions[0]))

plt.figure(figsize=(10,10))
for i in range(25):
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(mnb.getBinaryByTag(test_image_path)[i] / 255, cmap=plt.cm.binary)
  plt.xlabel(np.argmax(predictions[i]))
plt.show()