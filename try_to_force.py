import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import load_and_guess

import time
import random

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
    self.fc1 = nn.Linear(1, 100)
    self.fc2 = nn.Linear(100, 200)
    self.fc3 = nn.Linear(200, 784)
  def forward(self, x):
    x = F.relu(self.fc1(x))
  
    x = F.relu(self.fc2(x))
    
    x = F.rrelu(self.fc3(x))
    
    
    x = x.view(-1, 1, 28, 28)
    
    return x

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device}")

true_values = 0
false_values = 0
report = "";

# open the image
directory = './testdata/'
for filename in os.listdir(directory):

    guess = load_and_guess.LAGmain(os.path.join(directory, filename))

    img = Image.open(os.path.join(directory, filename)).convert('L').resize((28, 28))
    numpy_byte_img = torch.tensor((1 - (np.array(img) / 255)).reshape(1, 1, 28, 28).astype('float32')).to(device)

    # now the training starts

    model_state_dict = torch.load('./saved_models_pytorch/saved_model.p')

    model_to_break = Net().to(device)

    model_to_break.load_state_dict(model_state_dict)

    goal_number = 1

    # start anti-training

    net = antiNet().to(device)

    criterion_binary = nn.BCELoss()
    criterion_image = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    item = torch.tensor([np.array([0], dtype='float32')]).to(device)

    real_number = guess
    
    print("Guess: ", real_number)
    
    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    position = random.randint(0,len(numbers)-1)  
    print("Position before: ", position)
    if position == guess:
        if position == 9:
            position = position - 1
        else:
            position = position + 1
    print("Random number: ", position)
    numbers[position] = 1


    for epocs in range(1):
      for i in trange(5000):
        # image check
        optimizer.zero_grad()
        
        outputs = net(item)
        
        image_loss = criterion_image(outputs, numpy_byte_img)
        
        image_loss.backward()
        optimizer.step()
      
        # noise adder
        optimizer.zero_grad()

        outputs = net(item)

        check_net = model_to_break(outputs)

        check_net_loss = criterion_binary(check_net[0], torch.tensor(numbers, dtype=torch.float32).to(device))
        
        check_net_loss.backward()
        optimizer.step()

    print(check_net_loss)
    test_of_system = net(item)

    output_test = test_of_system

    check_of_system = model_to_break(test_of_system)

    print(check_of_system)
    
    image_array_data_guess = np.argmax(check_of_system.cpu().detach().numpy()[0])

    pyplot.imshow(output_test.cpu().detach().numpy()[0][0], cmap='Greys')
    pyplot.xlabel(image_array_data_guess)
    pyplot.savefig(f"./supersave/{time.time()}_{image_array_data_guess}_plot.png")
    print("\n")
    
    fooled = guess != image_array_data_guess
    
    if not fooled:
        true_values = true_values+1
    else:
        false_values = false_values+1
        
    report = report + "File: " + filename + "\n"
    report = report + "Guess of original image: " + str(guess) + "\n"
    report = report + "Guess of adversarial image: " + str(image_array_data_guess) + "\n"
    report = report + "Accurately fooled? " + str(fooled) + "\n\n"

report = "Accuracy of adversarial training: " + str(false_values / (false_values + true_values)*100) + "\n\n" + report

text_file = open(f"./supersave/{time.time()}_REPORT.txt", "w")
n = text_file.write(report)
text_file.close()

print(report)