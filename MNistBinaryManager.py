import numpy as np
import struct

class MNistBinaryManager:
  __mnist_loaded_binaries = {}

  def getBinaryByTag(self, tag):
    return self.__mnist_loaded_binaries[tag]

  # loads in an idx file 
  # return a numpy array based on the rows and columns
  def load_idx_file(self, tlp):
    # open the file and read in the contents
    fileContent = None

    with open(tlp, mode='rb') as file:
      fileContent = file.read()

    # first unpack the magic number to detect what kind of file it is
    # first two bytes of the magix number should be 0
    # third depends on the type of the data
    '''
    0x08: unsigned byte
    0x09: signed byte
    0x0B: short (2 bytes)
    0x0C: int (4 bytes)
    0x0D: float (4 bytes)
    0x0E: double (8 bytes)
    '''
    
    # 4th byte identifies dimensions for the data system

    # firstly read in the magic number in 4 bytes because it is 32 bits big endian
    # I chose to decode as an unsigned short (2 bytes) and 2 bytes of length 1
    
    unpacked_magic_number = struct.unpack(">HBB", fileContent[0:4])
    
    # throws an error if the magic number parity bytes are wrong
    assert unpacked_magic_number[0] == 0
    
    # the last 2 bytes of the magic number contain the operation being performed and how many dimensions the matrix of data has
    data_unit_length = unpacked_magic_number[1] # the length of one piece of data
    dimensions = unpacked_magic_number[2] # how many dimensions to look for
    
    # read in the dimension parameters into a variable
    
    unpacked_dimension_integers_32 = struct.unpack(">" + ("I") * dimensions, fileContent[4:(4 * dimensions + 4)])
    
    #print(unpacked_dimension_integers_32)
    
    file_content_without_header = fileContent[(4 + 4 * dimensions):]
    
    numpy_byte_array = np.frombuffer(file_content_without_header, dtype='uint8').reshape(unpacked_dimension_integers_32)

    return numpy_byte_array

  def __init__(self, training_label_path, training_image_path, test_label_path, test_image_path):
    self.__mnist_loaded_binaries[training_label_path] = self.load_idx_file(training_label_path)
    self.__mnist_loaded_binaries[training_image_path] = self.load_idx_file(training_image_path)
    self.__mnist_loaded_binaries[test_label_path] = self.load_idx_file(test_label_path)
    self.__mnist_loaded_binaries[test_image_path] = self.load_idx_file(test_image_path)
    
    #print(self.mnist_loaded_binaries)
    #print("DONE")