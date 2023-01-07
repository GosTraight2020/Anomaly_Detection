import numpy as np
import torch
from torch.nn import Module
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
from time import time

def make_condition_label(size):

    labels = []
    for i in range(size[0]):
        for j in range(size[1]):
            labels.append(i)

    labels = np.array(labels)
    labels = torch.from_numpy(labels)
    labels = torch.nn.functional.one_hot(labels, num_classes=10)
    labels = labels.float()
    return labels

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def log(info, level=1):
    if level == 1:
        print('[INFO] {}'.format(info))
    elif level == 2:
        print('[DEBUG] {}'.format(info))
    elif level == 3:
        print('------------------------------------------')
        print('[ATTENTION] {}'.format(info))
        print('------------------------------------------')



def evaluation(model, dataloader):
    labels = []
    preds = []
    model.eval()
    with torch.no_grad():
        for imgs, masks, _ in tqdm(dataloader):
            imgs = imgs.float()
            labs = masks.view((masks.shape[0], -1)).max(1)[0].tolist()

            logits = model(imgs)

            logits = torch.mean((imgs-logits)**2, (1,2,3))

            labels += labs
            preds += logits.cpu().numpy().tolist()

    return roc_auc_score(labels, preds)



