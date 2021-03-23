import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

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

def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"

  print(f"Using {device}")

  trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2, pin_memory=1)

  net = Net().to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
  
  for epoch in range(3):
    accuracy = 0
    for i, (item, label) in enumerate(tqdm(trainloader), 0):
      item, label = item.to(device), label.to(device)

      optimizer.zero_grad()

      outputs = net(item)
      loss = criterion(outputs, label)
      loss.backward()
      optimizer.step()
      
      accuracy += (outputs.argmax(dim=1) == label).int().sum().item()

    print(f"Accuracy for epoch {epoch}: %{accuracy / 60000 * 100}")
    
  
  torch.save(net.state_dict(), './saved_models_pytorch/saved_model.p')

if __name__ == "__main__":
  main()