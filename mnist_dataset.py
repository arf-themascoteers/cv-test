from torch.utils.data import Dataset
import os
import cv2
import torch
import matplotlib.pyplot as plt

class MNIST(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        self.data_indices = None
        self.folder = "data/test"
        if is_train:
            self.folder = "data/train"

        self.data_indices = [i for i in os.listdir(self.folder)]

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        value = self.data_indices[idx]
        digit = int(value[0])
        image = cv2.imread(self.folder+"/"+value)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return torch.tensor(image,dtype=torch.float),torch.tensor(digit)

if __name__ == "__main__":
    m = MNIST(False)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(m, batch_size=1, shuffle=False)
    for data, y in dataloader:
        print(type(data))
        print(data.shape)
        print(y)
        break
