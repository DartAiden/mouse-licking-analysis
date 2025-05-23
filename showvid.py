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
import cv2 as cv
import isstim
import os

class createDataset(Dataset):
    def __init__(self, data, labels):
        self.labels = pd.read_csv(labels)
        self.data = data
        self.transform = transforms.Compose([
        transforms.CenterCrop(size = 400), 
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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(.1)

        self.conv2 = nn.Conv2d(6, 16, 5)
        self.dropout2 = nn.Dropout(.1)
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
PATH = './licktrain4.pth'
net= Net()
net.load_state_dict(torch.load(PATH, weights_only=True))
net.eval()

lickpoints = []
imtransfor = transforms.Compose([
        transforms.CenterCrop(size = 400), 
        transforms.Resize((224, 224)),
    ])
font = cv.FONT_HERSHEY_SIMPLEX
org = (00, 185)
fontscale = 1
color= (0,0,0)
thickness = 2

count = 0


title = "annotatedvid.mp4"
count +=1
full_path =r"inputvids\T51_250429_lick_Video_CROP.mp4"

cap = cv.VideoCapture(full_path)
fourcc = cv.VideoWriter_fourcc(*"XVID")
fps = int(cap.get(cv.CAP_PROP_FPS))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
writer = cv.VideoWriter(title, fourcc,fps, (width, height))



initialize = False

with torch.no_grad():
    ret = True
    while ret:
        ret, img = cap.read()
        if not ret or img is None:
            break
        if not initialize:
            stim = False
            stimmer = isstim.Stimmer(img)
            initialize = True
        else:
            stim = stimmer.isStim(img)
        disp = img
        #ts = (cap.get(cv.CAP_PROP_POS_MSEC))
        stimorg = (0, 320)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = img.float() / 255.0
        imga = imtransfor(img)
        imga = imga.unsqueeze(0) 
        temp = net(imga)
        _, predicted = torch.max(temp, 1)
        #print(predicted)
        pred = predicted.item()
        #print(ts)
        if pred  == 0:
            text = "NOT LICKING"
        else:
            text = "LICKING"
        if stim:
            stimtext = "STIMMING"
        else:
            stimtext = "NOT STIMMING"
        disp = cv.putText(disp, text, org, font, 
                fontscale, color, thickness, cv.LINE_AA)
        disp = cv.putText(disp, stimtext, stimorg, font, 
                fontscale, color, thickness, cv.LINE_AA)
        cv.imshow('Licking',disp)
        writer.write(disp)
        if cv.waitKey(1) == ord('q'):
            break
    print("DONE")
    cap.release()
    writer.release()
    cv.destroyAllWindows()
