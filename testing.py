import os
import pandas as pd
from torch.utils.data import Dataset,DataLoader
from torchvision.io import read_image,read_video,VideoReader
from torchvision.transforms import v2
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from custom_dataset import CustomVidDataset
from torchvideotransforms import video_transforms, volume_transforms
import torch
from torch import nn

video_transform_list = [
			video_transforms.Resize((224, 224)),
            volume_transforms.ClipToTensor()]
video_transform_list = [MViT_V2_S_Weights.KINETICS400_V1.transforms()]
transforms = video_transforms.Compose(video_transform_list)

# data_load = CustomImageDataset('img/label.csv','img/')
data_vid_load = CustomVidDataset(annotations_file = 'data/test_label.csv',vid_dir = 'data/test',transform=transforms)

data_loader = DataLoader(
    data_vid_load,
    batch_size=30,
    shuffle=True
)

for batch,X in data_loader:
    print(X.shape)
# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200*200, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 12),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(data_loader, model, loss_fn, optimizer)
print("Done!")