import torch
import torchvision
import torchvision.transforms as transforms
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class createDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = pd.read_csv(labels)
        self.data = data
        self.transform = transforms.Compose([
        transforms.CenterCrop(size = 400), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation = .1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.Resize((224, 224)),
        
    ])
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.data, self.labels.iloc[idx, 0])
        image = read_image(img_path).float() / 255.0 
        label = int(self.labels.iloc[idx,1])
        image = self.transform(image)
        return image,label

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.dropout = .5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(self.dropout)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout2 = nn.Dropout(self.dropout)
        dummy_input = torch.zeros(1, 3, 224, 224)

        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)

        self.flattened_size = x.view(-1).shape[0]
        self.fc1 = nn.Linear(self.flattened_size, 240)
        self.fc2 = nn.Linear(240, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

net = Net()

licklabels = 'annotations.csv'
lickdir = r'C:\Users\adart\Documents\mouse-licking-analysis\licking_combined'
lickdataset = createDataset(lickdir, licklabels)


lick_dataloader = DataLoader(lickdataset, batch_size=32, shuffle=True)
dataiter = iter(lick_dataloader)
images, labels = next(dataiter)
imlabels = []
#for _, labels in lick_dataloader:
#    imlabels.extend(labels.numpy())
#class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(imlabels), y=imlabels)
#class_weights = torch.tensor(class_weights, dtype=torch.float)
#print(' '.join(f'{labels[j]}' for j in range(64)))
#imshow(torchvision.utils.make_grid(images))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)


for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(lick_dataloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    correct = 0
    total = 0
    class_counts = [0, 0]
    with torch.no_grad():
        for data in lick_dataloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for p in predicted:
                class_counts[p] += 1

    print(f'Accuracy: {100 * correct / total}% on cycle {epoch}')
    print(f'Predicted class distribution: {class_counts}')

PATH = './licktrain5.pth'
torch.save(net.state_dict(), PATH)


net.load_state_dict(torch.load(PATH, weights_only=True))
correct = 0
total = 0
count = 0
with torch.no_grad():
    for data in lick_dataloader:
        if count < 1000:
            count +=1
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')