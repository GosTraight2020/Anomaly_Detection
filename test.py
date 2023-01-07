import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from load_dataset import LandslideDataSet
from utils import evaluation
from parameters import arg

import os
from tqdm import tqdm
import numpy as np
import cv2

def norm(inputs):
    tmp = (inputs-inputs.min()) / (inputs.max()-inputs.min())
    tmp = np.uint8(tmp * 255)
    return tmp

def test():
    evaldata = LandslideDataSet(train=False)
    evalloader = DataLoader(evaldata, batch_size=1, num_workers=0, shuffle=False)

    model = torch.load('./checkpoints/AE/landslide_128/model.pth')
    model.eval()

    visual_dir = './visual'
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    with torch.no_grad():
        for img, mask, filepath in tqdm(evalloader):
            img = img.cpu().float()
            mask = mask[0].numpy()

            logits = model(img)

            mse = torch.mean((img - logits) ** 2, dim=1)[0].cpu().numpy()

            mask = norm(mask)
            mse = norm(mse)

            filename = filepath[0].split('/')[-1].split('.')[0]

            cv2.imwrite(os.path.join(visual_dir, filename+'_mask.jpg'), mask)
            cv2.imwrite(os.path.join(visual_dir, filename+'_map.jpg'), mse)





if __name__=='__main__':
    test()




