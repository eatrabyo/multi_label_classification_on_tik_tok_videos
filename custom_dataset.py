import pandas as pd
import os
from torchvision.io import read_video,read_video_timestamps
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import numpy as np

class CustomVidDataset(VisionDataset):
    def __init__(self, annotations_file, vid_dir, transform= None, target_transform=None):
        self.vid_labels = pd.read_csv(annotations_file)
        self.vid_dir = vid_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.vid_labels)

    def __getitem__(self, idx):

        vid_path = os.path.join(self.vid_dir, self.vid_labels.iloc[idx, 0])
        vid = read_video(vid_path,pts_unit='sec',end_pts=5.0)
        label = self.vid_labels.iloc[idx, 5:17]
        array_label = label.to_numpy()
        array_label = array_label.astype(int)
        trans_vid = vid[0][:100]
        if self.transform:
            trans_vid = self.transform(trans_vid)
        if self.target_transform:
            label = self.target_transform(array_label)
        return trans_vid, array_label
    
if __name__ == "__main__":
    data_vid_load = CustomVidDataset(annotations_file = 'data/test_label.csv',vid_dir = 'data/test')

    x,y = data_vid_load.__getitem__(6)

    t = read_video_timestamps('data/test/Beauty_7.mp4',pts_unit='sec')
    print(x.shape)
    print(len(t[0]))
    # print(t)
    print('dd')