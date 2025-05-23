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


"""
This script is used for processing the videos. 
"""

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
PATH = './licktrain5.pth'
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


for a in os.listdir('inputvids'):
    title = a.replace('.mp4','_annotated.mp4')
    full_path = os.path.join('inputvids', a)
    cap = cv.VideoCapture(full_path)
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter(title, fourcc,fps, (width, height)) #Open up new video to save the annotated videos
    initialize = False
    print(f"NOW PROCESSING {a}")
    with torch.no_grad():
        times = [] #Record times for all frames for lick and stimuli - these two lists should be the same
        licks = []
        stims = [] #Record whether 
        ret = True
        while ret:
            ret, img = cap.read()
            if not ret or img is None:
                break
            if not initialize:
                stim = False
                stimmer = isstim.Stimmer(img)
                initialize = True #Extract the blue channel for benchmark
            else:
                stim = stimmer.isStim(img) #Evaluate
            disp = img
            ts = (cap.get(cv.CAP_PROP_POS_MSEC))/1000 #Extract the timestamp in milliseconds
            stimorg = (0, 320)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB) #Rearrange the color channels
            img = torch.from_numpy(img) #Turn the numpy array into a torch tensor
            img = img.permute(2, 0, 1) #Rearrange all channels
            img = img.float() / 255.0 #Normalize
            imga = imtransfor(img) #Perform the relevant transformations
            imga = imga.unsqueeze(0) #Add a new dimension to position 0, to have it be a batch of one
            temp = net(imga) #Insert it into the neural network
            _, predicted = torch.max(temp, 1) #Find the predicted value
            pred = predicted.item() #Extract it
            if pred  == 0:
                text = "NOT LICKING" #annotate the frame appropriately
                licks.append(0)
            else:
                text = "LICKING"
                licks.append(1)
            if stim:
                stimtext = "STIMMING"
                stims.append(1)
            else:
                stimtext = "NOT STIMMING"
                stims.append(0)
            times.append(ts)
            disp = cv.putText(disp, text, org, font, #Insert the annotation over the image
                    fontscale, color, thickness, cv.LINE_AA)
            disp = cv.putText(disp, stimtext, stimorg, font, 
                    fontscale, color, thickness, cv.LINE_AA)
            #cv.imshow('Licking',disp)
            writer.write(disp)
            #if cv.waitKey(1) == ord('q'):
            #    break
        print(f"DONE WITH {a}")
        cap.release()
        writer.release()
        cv.destroyAllWindows()
    completedf = pd.DataFrame({"Times" : times, "Lick_Signal" : licks, "Stim_Signal" : stims}) #Write out the file
    completetitle = a.replace('.mp4','_complete.csv')
    completedf.to_csv(completetitle, index=False)