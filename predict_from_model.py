import sys
import os
import numpy

from tensorflow import keras
from tensorflow.keras.models import load_model

# debugging for image reading
from PIL import Image
from matplotlib import pyplot

# check if an argument was supplied
assert len(sys.argv) > 1

# check if the argument passed was a file
assert os.path.isfile(sys.argv[1])

model = load_model("savedmodel/trainedmnistdata.h5")

# convert image into a numpy byte array normalized at 1
# max image size is 28x28 cuz that is the size of the mnist image set
img = Image.open(sys.argv[1]).convert('L').resize((28, 28))

# this is 1- because normally 255 is white, but here 255 is black
numpy_byte_img = (1 - (numpy.array(img) / 255)).reshape(1, 28, 28)

#print(numpy_byte_img)

#print(numpy_byte_img)

#pyplot.imshow(numpy_byte_img)
#pyplot.show()

print(model.predict_classes(numpy_byte_img))