import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys
import numpy as np
from PIL import Image
from matplotlib import pyplot
from tqdm import trange

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
    
    x = self.soft(x)
    
    #print(x.shape)
    
    return x

class antiNet(nn.Module):
  def __init__(self):
    super(antiNet, self).__init__()
    self.fc1 = nn.Linear(10, 100)
    self.fc2 = nn.Linear(100, 200)
    self.fc3 = nn.Linear(200, 784)
  def forward(self, x):
    x = F.relu(self.fc1(x))
  
    x = self.fc2(x)
    
    x = F.relu(self.fc3(x))
    
    x = F.relu(x.view(-1, 1, 28, 28))
    
    #print(x.shape)
    return x

# file to corrupt
assert len(sys.argv) > 1
assert os.path.isfile(sys.argv[1])

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device}")

# open the image

img = Image.open(sys.argv[1]).convert('L').resize((28, 28))
numpy_byte_img = torch.tensor((1 - (np.array(img) / 255)).reshape(1, 1, 28, 28).astype('float32')).to(device)

# now the training starts

model_state_dict = torch.load('./saved_models_pytorch/saved_model.p')

model_to_break = Net().to(device)

model_to_break.load_state_dict(model_state_dict)

#image_generate = image_generate_raw.reshape(28, 28)

goal_number = 1

# start anti-training

net = antiNet().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

item = torch.tensor([np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype='float32')]).to(device)

for epocs in range(5):
  for i in trange(500):
    optimizer.zero_grad()

    outputs = net(item) + numpy_byte_img

    check_net = model_to_break(outputs)

    loss = criterion(check_net[0], torch.tensor([0, 0, 30, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32).to(device))
    #print(loss)
    loss.backward()
    optimizer.step()


test_of_system = net(item)

output_test = test_of_system + numpy_byte_img

check_of_system = model_to_break(test_of_system)

print(check_of_system)

pyplot.imshow(output_test.cpu().detach().numpy()[0][0], cmap='Greys')
pyplot.xlabel(np.argmax(check_of_system.cpu().detach().numpy()[0]))
pyplot.show()

#print(test_of_system)
#

#print(loss)

#print(check_net > 24)

#outputs = net(check_net)

#print(check_net)

#loss = criterion(outputs, label)
#loss.backward()
#optimizer.step()