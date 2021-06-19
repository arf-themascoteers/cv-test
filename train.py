from simple_net import SimpleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from mnist_dataset import MNIST

def train():
    NUM_EPOCHS = 500
    BATCH_SIZE = 1000

    working_set = MNIST(is_train=True)

    dataloader = DataLoader(working_set, batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch  in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            data = data.reshape(data.shape[0], 1, data.shape[1],data.shape[2])
            optimizer.zero_grad()
            y_pred = model(data)
            loss = F.nll_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    torch.save(model.state_dict(), 'models/lin.h5')
    return model

train()


