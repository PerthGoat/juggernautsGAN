from MNistBinaryManager import MNistBinaryManager
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

# now set up keras
# creates a Keras tensor
inputs = keras.Input(shape=keras_shape, name="mnist_images")

# creates a callable layer function
layer = layers.Dense(32, activation='relu')

# the outputs after putting the inputs through the training layer
outputs = layer(inputs)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
  optimizer=keras.optimizers.RMSprop(),
  loss=keras.losses.SparseCategoricalCrossentropy(),
  metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

history = model.fit(
mnb.getBinaryByTag(training_image_path),
mnb.getBinaryByTag(test_image_path),
batch_size=64,
epochs=2)'''