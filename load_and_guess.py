import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy
from PIL import Image

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 5, 3) #convolutional layer
    self.fc1 = nn.Linear(13, 100) #dense layers
    self.fc2 = nn.Linear(100, 10)
    self.fc3 = nn.Linear(650, 10)
  def forward(self, x):
    #print(f"orig: {x.shape}")
    x = F.relu(self.conv1(x))
    #print(x.shape)
    
    x = F.max_pool2d(x, (2, 2)) #pooling layer
    
    #print(x.shape)
    
    x = self.fc1(x)
    
    #print(x.shape)
    
    x = self.fc2(x)
    
    #print(x.shape)
    
    x = x.view(-1, 650)
    
    #print(x.shape)
    
    x = F.relu(self.fc3(x))
    
    #print(x.shape)
    
    return x

assert len(sys.argv) > 1
assert os.path.isfile(sys.argv[1])

model_state_dict = torch.load('./saved_models_pytorch/saved_model.p')

model = Net().double()

model.load_state_dict(model_state_dict)

img = Image.open(sys.argv[1]).convert('L').resize((28, 28))

numpy_byte_img = (1 - (numpy.array(img) / 255)).reshape(1, 1, 28, 28)

print(numpy.argmax(model(torch.from_numpy(numpy_byte_img)).detach().numpy()))