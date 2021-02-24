# imports for mnist binary handling
import struct
import numpy as np
import matplotlib.pyplot as plt

# imports for tensorflow stuff
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# read binary in
# file name of the IDX file to load
FILE_NAME = "train-images.idx3-ubyte"
# name of the directory to put the converted images
IMAGE_SAVE_DIRECTORY = "exported_images/"

# open the file and read in the contents
fileContent = None

with open(FILE_NAME, mode='rb') as file:
  fileContent = file.read()

# unpack the first 4 32-bit integers that are the header. They are big endian
unpacked_data = struct.unpack(">IIII", fileContent[0:16])

# label the extracted integers
MAGIC_NUMBER = unpacked_data[0]
NUMBER_OF_IMAGES = unpacked_data[1]
NUMBER_OF_ROWS = unpacked_data[2]
NUMBER_OF_COLUMNS = unpacked_data[3]

# make sure the magic number matches the number for this file on the mnist website
assert MAGIC_NUMBER == 2051

# trim the header off of the file for further operations
file_content_without_header = fileContent[16:]

# convert the byte array into a numpy array of 8-bit integers, reshape it to 60000 images with width and height from the header
numpy_byte_array = (np.frombuffer(file_content_without_header, dtype='uint8')).reshape(NUMBER_OF_IMAGES, NUMBER_OF_ROWS * NUMBER_OF_COLUMNS)

# now set up keras
# creates a Keras tensor
inputs = keras.Input(shape=(NUMBER_OF_ROWS*NUMBER_OF_COLUMNS,), name="mnist_images")

# creates a callable layer function
layer = layers.Dense(32, activation='relu')

# the outputs after putting the inputs through the training layer
outputs = layer(inputs)

model = keras.Model(inputs=inputs, outputs=outputs)