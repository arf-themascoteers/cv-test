from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import cv2
import numpy as np
import time

working_set = torchvision.datasets.MNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)

dataloader = DataLoader(working_set, batch_size=1, shuffle=False)

for data, y_true in dataloader:
    arr = data[0][0].numpy()*255
    arr = cv2.resize(arr, (7,7))
    epoch_time = int(time.time())
    file = str(y_true.item()) +"_"+str(epoch_time)+".jpg"
    cv2.imwrite("data/train/"+file, arr)


