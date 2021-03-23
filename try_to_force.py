import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

import os
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot
from tqdm import trange
from torch.autograd import Variable

import cv2

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
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

class antiNet(nn.Module):
  def __init__(self):
    super(antiNet, self).__init__()
    self.fc1 = nn.Linear(784, 784 * 2)
    #self.fc2 = nn.Linear(100, 200)
    self.fc3 = nn.Linear(784 * 2, 784)
    self.soft = nn.Softmax(dim=2)
  def forward(self, x):
    x = torch.rrelu(torch.rrelu(self.fc1(x)))
  
    #x = F.rrelu(self.fc3(x))
    
    x = torch.sigmoid(torch.sigmoid(self.fc3(x)))
    
    x = x.view(-1, 1, 28, 28)
    
    x = self.soft(x)
    
    return x

# file to corrupt
assert len(sys.argv) > 1
assert os.path.isfile(sys.argv[1])

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device}")

# open the image

img = Image.open(sys.argv[1]).convert('L').resize((28, 28))
numpy_byte_img = torch.tensor((1 - (np.array(img) / 255)).flatten().astype('float32')).to(device)
#numpy_byte_img = torch.tensor((1 - (np.array(img) / 255)).reshape(1, 1, 28, 28).astype('float32')).to(device)

# now the training starts

model_state_dict = torch.load('./saved_models_pytorch/saved_model.p')

model_to_break = Net().to(device)

model_to_break.load_state_dict(model_state_dict)

# start anti-training

net = antiNet().to(device)

#net.apply(init_weights)

criterion_binary = nn.MSELoss()
criterion_image = nn.MSELoss()
mnloss = nn.LogSoftmax(dim=2)
optimizer = optim.Adam(net.parameters(), lr=0.01)

for epocs in range(1):
  for i in trange(5000):
    # image check
    '''optimizer.zero_grad()
    
    outputs = net(item)
    
    image_loss = criterion_image(outputs, numpy_byte_img)
    
    image_loss.backward()
    optimizer.step()'''
  
    # noise adder
    optimizer.zero_grad()

    outputs = net(numpy_byte_img)
    
    image_loss = criterion_image(mnloss(outputs), numpy_byte_img.reshape(1, 1, 28, 28))
    
    #check_net = model_to_break(outputs)

    #check_net_loss = criterion_binary(check_net[0], torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=torch.float32).to(device))
    
    image_loss.backward()
    optimizer.step()

print(image_loss)
test_of_system = net(numpy_byte_img)

output_test = test_of_system

check_of_system = model_to_break(output_test)

print(check_of_system)

image_array_data_numpy = output_test.cpu().detach().numpy()[0][0]
#image_array_data_numpy = (1 - output_test.cpu().detach().numpy()[0][0]) * 255

image_array_data_guess = np.argmax(check_of_system.cpu().detach().numpy()[0])

#print(image_array_data_guess)

#print(image_array_data_numpy)

#img = Image.fromarray(image_array_data_numpy).convert("L")

#img.save(f"./supersave/{time.time()}_{image_array_data_guess}.bmp")
#img.show()

pyplot.imshow(image_array_data_numpy, cmap='Greys')
pyplot.xlabel(image_array_data_guess)
pyplot.show()