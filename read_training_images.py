import struct
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# file name of the IDX file to load
FILE_NAME = "train-images-idx3-ubyte"
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

# convert the byte array into a numpy array of 8-bit integers, subtract 255 by each to invert the color palette, reshape it to 60000 images with width and height from the header
numpy_byte_array = (255 - np.frombuffer(file_content_without_header, dtype='uint8')).reshape(NUMBER_OF_IMAGES, NUMBER_OF_ROWS, NUMBER_OF_COLUMNS)

#PIL_image = Image.fromarray(numpy_byte_array[0], 'P')

#plt.imshow(PIL_image)
#plt.show()

# loop through each loaded image and write it to a PNG file (tqdm is a progress bar library)
for i in tqdm(range(NUMBER_OF_IMAGES)):
  the_image = Image.fromarray(numpy_byte_array[i], 'P')
  the_image.save("{}{}.PNG".format(IMAGE_SAVE_DIRECTORY, i)); 
