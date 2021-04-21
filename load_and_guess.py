import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy
from PIL import Image

class LAGNet(nn.Module):
  def __init__(self):
    super(LAGNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 5, 3) #convolutional layer
    self.fc1 = nn.Linear(13, 100) #dense layers
    self.fc2 = nn.Linear(100, 10)
    self.fc3 = nn.Linear(650, 10)
    self.soft = nn.Softmax(dim=1)
  def forward(self, x):
    x = F.relu(self.conv1(x))
    
    x = F.max_pool2d(x, (2, 2)) #pooling layer
    
    x = self.fc1(x)
    
    x = self.fc2(x)
    
    x = x.view(-1, 650)
    
    x = F.relu(self.fc3(x))
    
    x = self.soft(x)
    
    return x

def LAGmain(pic_location):

    model_state_dict = torch.load('./saved_models_pytorch/saved_model.p')

    model = LAGNet()

    model.load_state_dict(model_state_dict)

    img = Image.open(pic_location).convert('L').resize((28, 28))

    numpy_byte_img = (1 - (numpy.array(img) / 255)).reshape(1, 1, 28, 28).astype('float32')

    return numpy.argmax(model(torch.from_numpy(numpy_byte_img)).detach().numpy())