import torch
from torchvision import datasets
from PIL import Image
import numpy as np
from torchvision import models
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import os
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision.transforms import *
from torch.utils.data import DataLoader
from matplotlib.ticker import MaxNLocator

num_classes = 10
batch_size = 16
num_of_workers = 3

DATA_PATH_TRAIN = '/Users/macbookpro/Documents/PythonProjects/Sign_Language_Digits/dataset_train_test/train'
DATA_PATH_TEST = '/Users/macbookpro/Documents/PythonProjects/Sign_Language_Digits/dataset_train_test/test'

train_transforms = transforms.Compose([
    transforms.Scale((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
test_transforms = transforms.Compose([
    transforms.Scale((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=DATA_PATH_TRAIN, transform=train_transforms)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_of_workers)

test_dataset = datasets.ImageFolder(root=DATA_PATH_TEST, transform=train_transforms)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_of_workers)

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(30976, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #print("num_flat_features : ", self.num_flat_features(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x,dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Net(nn.Module):

    # This constructor will initialize the model architecture
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            # Putting a 2D Batchnorm after CNN layer
            nn.BatchNorm2d(32),
            # Adding Relu Activation
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            # Adding Dropout
            nn.Dropout(p=0.5),
            #nn.Linear(32 * 32 * 32, 512),
            nn.Linear(30000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

    # Defining the forward pass
    def forward(self, x):
        print("num_flat_features : ", self.num_flat_features(x))
        # Forward Pass through the CNN Layers
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        # Forwrd pass through Fully Connected Layers
        x = self.linear_layers(x)
        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# get some random training images
#dataiter = iter(train_loader)
#images, labels = dataiter.next()
#print(labels[1])
#plt.imshow(np.transpose(images[1].numpy(), (1, 2, 0)))
#plt.show()


train_loss = []
train_accuracy = []
valid_loss = []
valid_accuracy = []

########################################
#       Training the model             #
########################################
def train(epoch, model, train_loader):
    model.train()
    exp_lr_scheduler.step()
    tr_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        #print("batch_idx : ", batch_idx)
        data, target = Variable(data, volatile=False), Variable(target)

        # Clearing the Gradients of the model parameters
        optimizer.zero_grad()
        output = model(data)
        pred = torch.max(output.data, 1)[1]
        correct += (pred == target).sum()
        total += len(data)

        # Computing the loss
        loss = criterion(output, target)

        # Computing the updated weights of all the model parameters
        loss.backward()
        optimizer.step()
        tr_loss = loss.item()
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t Accuracy: {} %'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * correct / total))
            torch.save(model.state_dict(), './model.pth')
            torch.save(model.state_dict(), './optimizer.pth')
    train_loss.append(tr_loss / len(train_loader))
    train_accuracy.append(100 * correct / total)


########################################
#       Evaluating the model           #
########################################
def evaluate(model, data_loader):
    model.eval()
    loss = 0
    correct = 0
    total = 0
    for data, target in data_loader:
        data, target = Variable(data, volatile=False), Variable(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)
        loss += F.cross_entropy(output, target, size_average=False).item()
        pred = torch.max(output.data, 1)[1]
        total += len(data)
        correct += (pred == target).sum()
    loss /= len(data_loader.dataset)
    valid_loss.append(loss)
    valid_accuracy.append(100 * correct / total)
    print('\nAverage Validation loss: {:.5f}\tAccuracy: {} %'.format(loss, 100 * correct / total))


n_epochs = 20
model = Net2()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
for epoch in range(n_epochs):
    train(epoch, model, train_loader)
    evaluate(model, test_loader)

plt.plot(range(1, len(train_loss)+1), train_loss, 'b-', label='training loss')
plt.plot(range(1, len(valid_loss)+1), valid_loss, 'g-', label='validation loss')
plt.plot(range(1, len(train_accuracy)+1), train_accuracy, 'r-', label='training accuracy')
plt.plot(range(1, len(valid_accuracy)+1), valid_accuracy, 'y-', label='validation accuracy')
plt.xlabel('epochs', fontsize=16)
plt.ylabel('loss', fontsize=16)

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.legend()
plt.show()
