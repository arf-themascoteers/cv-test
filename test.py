from simple_net import SimpleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from mnist_dataset import MNIST

def test():
    BATCH_SIZE = 100

    working_set = MNIST(is_train=False)

    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleNet()
    model.load_state_dict(torch.load("models/lin.h5"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, y_true in dataloader:
            data = data.reshape(data.shape[0], 1, data.shape[1],data.shape[2])
            y_pred = model(data)
            pred = torch.argmax(y_pred, dim=1, keepdim=True)
            correct += pred.eq(y_true.data.view_as(pred)).sum()
            total += 1

    print(f"{correct} correct among {len(working_set)}")

test()