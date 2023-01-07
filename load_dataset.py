# import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os, glob
import h5py
from utils import log
import random
from parameters import arg

class LandslideDataSet(Dataset):
    def __init__(self, train=True, transform=None):
        self.mean = [-0.4914, -0.3074, -0.1277, -0.0625, 0.0439, 0.0803, 0.0644, 0.0802, 0.3000, 0.4082, 0.0823, 0.0516, 0.3338, 0.7819]
        self.std = [0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354, 0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913]
        self.transform = transform
        self.channels = [3, 2, 1]
        self.img_dir = './data/landslide'

        if train:
            self.image_pattern = os.path.join(self.img_dir, 'train', 'image*.h5')
            self.image_files = glob.glob(self.image_pattern)
        else:
            self.image_pattern = os.path.join(self.img_dir, 'test', 'image*.h5')
            self.image_files = glob.glob(self.image_pattern)

        log('Dataset has been created! Train : {}, Length :{}'.format(train, len(self.image_files)))

                
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        mask_file = image_file.replace('image', 'mask')
        with h5py.File(image_file, 'r') as image_f, h5py.File(mask_file, 'r') as mask_f:
            image_origin = image_f['img'][:]
            mask = mask_f['mask'][:]
        image = np.array(image_origin)
        mask = np.array(mask)

        for i in range(len(self.mean)):
            image[:,:, i] -= self.mean[i]
            image[:,:, i] /= self.std[i]

        waves = []
        for ch in self.channels:
            waves.append(image[:, :, ch-1:ch].copy())
        images = np.concatenate(waves, axis=-1)
        images[np.isnan(images)] = 0.000001
        images = np.transpose(images, axes=(2, 0, 1))


        if self.transform is not None:
            images = self.transform(images).float()

        return images, mask, image_file

if __name__ == '__main__':
    dataset = LandslideDataSet(train=False)
    dataloader = DataLoader(
        dataset,
        batch_size=arg.batch_size,
        num_workers=0,
        shuffle=True
    )

    for batch_img, mask, image_file in dataloader:
        print(batch_img.max(), batch_img.shape)
    