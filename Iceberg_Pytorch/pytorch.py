import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from termcolor import *
import os
from Data.constant import PYTORCH_MODEL_PATH, TRAIN_PATH, VAL_PATH, check_file, PYTORCH_OUTPUT_PATH_ROOT

check_file(TRAIN_PATH)
check_file(VAL_PATH)

if not os.path.exists(PYTORCH_OUTPUT_PATH_ROOT):
    os.makedirs(PYTORCH_OUTPUT_PATH_ROOT)

# prepare data
train_dir = TRAIN_PATH
validation_dir = VAL_PATH

transform_tensor = transforms.Compose([
        transforms.RandomResizedCrop(75),
        transforms.ToTensor(),
    ])

train_datasets = datasets.ImageFolder(os.path.join(train_dir), transform=transform_tensor)

val_datasets = datasets.ImageFolder(os.path.join(validation_dir), transform=transform_tensor)

print(len(train_datasets))

num_epochs = 1
batch_size = 32
learning_rate = 0.0003

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_datasets, batch_size=batch_size, shuffle=False,  num_workers=4)

# CNN Model (4 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64,32),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(32,16),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(16,2),
            nn.Softmax(),
        )


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

cnn = CNN()
# using GPU
# cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

train_loss_epoch = []
accuracy_epoch = []

# Train the Model
for epoch in range(num_epochs):
    loss_mini = []
    correct_epoch = []
    for i, (images, labels) in enumerate(train_loader):
        total = 0
        correct = 0
        #images = Variable(images).cuda()
        images = Variable(images)
        target = labels
        #labels = Variable(labels).cuda()
        labels = Variable(labels)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_mini.append(loss.data[0])
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #correct += (predicted == target.cuda()).sum()
        correct += (predicted == target).sum()
        correct_epoch.append(100*correct/total)

        if (i + 1) % 10 == 0:\
            print(colored('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f Accu: %d %%'\
                % (epoch + 1, num_epochs, i + 1, len(train_datasets) // batch_size, loss.data[0],100 * correct / total),"grey"))

    accuracy_epoch.append((np.array(correct_epoch).mean()))
    train_loss_epoch.append((np.array(loss_mini).mean()))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var)

correct = 0
total = 0

for images, labels in val_loader:
    #images = Variable(images).cuda()
    images = Variable(images)

    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    #correct += (predicted == labels.cuda()).sum()
    correct += (predicted == labels).sum()
print(colored('Test Accuracy of the model on the %d test images: %d %%' % (len(val_datasets),100 * correct / total), "red"))

# Save the Trained Model
torch.save(cnn.state_dict(), PYTORCH_MODEL_PATH)

plt.figure(0)
plt.plot(np.arange(len(train_loss_epoch)), np.array(train_loss_epoch))
plt.xlabel('Number of Iteration')
plt.ylabel('Training Loss Values')
plt.title('Training Loss of each epoch')

plt.figure(1)
plt.plot(np.arange(len(accuracy_epoch)), np.array(accuracy_epoch))
plt.xlabel('Number of Iteration')
plt.ylabel('Training Accuracy percentage')
plt.title('Training Accuracy of each epoch')

plt.show()
