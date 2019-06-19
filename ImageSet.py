import os
import cv2
from Rescale import Rescale
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os.path as osp

class ImageSet(Dataset):
    def __init__(self, folder_path, resize=None):
        files = os.listdir(folder_path)
        self.images = []
        for file in files:
            if not os.path.isdir(file):
                tmp_image = cv2.imread(osp.join(folder_path,file))
                self.images.append(tmp_image)

        if resize is not None:
            rescaler = Rescale(resize)
            for i in range(len(self.images)):
                self.images[i] = rescaler(self.images[i])

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)



if __name__ == '__main__':
    img0 = cv2.imread('./data/airplane/0029.jpg')
    img1 = cv2.imread('./data/airplane/0042.jpg')
    rescale = Rescale((224, 224))
    img0, img1 = rescale(img0), rescale(img1)

    print("Over...")
