import torch
import torch.utils.data as data
import glob2
import os
from PIL import Image
import numpy as np

class GIANA(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        org_dir = os.path.join(self.root, 'Original')
        seg_dir = os.path.join(self.root, 'GT')

        self.orgs = []
        self.segs =[]

        if self.train:
            self.orgs = glob2.glob(os.path.join(org_dir,'*.bmp'))
            self.segs = glob2.glob(os.path.join(seg_dir,'*.bmp'))
        else:
            self.orgs = glob2.glob(os.path.join(org_dir,'*.bmp'))
            self.segs = glob2.glob(os.path.join(seg_dir,'*.bmp'))

    def __getitem__(self, index):
        img = np.array(Image.open(self.orgs[index]).convert('RGB'))
        target = np.array(Image.open(self.segs[index]).convert('RGB'))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.orgs)


