from load_dataset import LandslideDataSet
import torch 
from torch.utils.data import DataLoader
from model import AutoEncoder
from torchvision import transforms
from torch import optim, autograd
import torchvision
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from utils import merge, log
import os
import scipy.misc
from parameters import arg
from utils import evaluation


def train():
    data_shape = [3, 128, 128]
    model = AutoEncoder('landslide', data_shape, C_lr=1e-4, G_lr=1e-4, summary_path='./summary/', checkpoint_path='./checkpoints')
    model.summary_writer.add_graph(model.ae, torch.randn(128, 3, 128, 128))
    train_dataset = LandslideDataSet(train=True)
    eval_dataset = LandslideDataSet(train=False)
    train_loader = DataLoader(train_dataset, batch_size=arg.batch_size, num_workers=0, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=arg.batch_size, num_workers=0, shuffle=False)
    
    num_per_epoch = len(train_loader)

    epochs = 1 if arg.debug else arg.num_epochs
    best_auc = 0.0
    aucs = []
    for epoch in range(arg.num_epochs):
        model.ae.encoder.train()
        model.ae.decoder.train()
        for i, (images, _, _) in enumerate(train_loader):
            step = num_per_epoch * epoch + i
            images = images.float()
            C_loss, W_dist, gp = model.train_critic_one_epoch(images)
            G_loss, recon_loss, loss, fake = model.train_generator_one_epoch(images)
            print('{}, l1_loss : {}'.format(step, recon_loss))
            model.summary_writer.add_scalar('loss/C_loss', C_loss, global_step=step)
            model.summary_writer.add_scalar('loss/G_loss', G_loss, global_step=step)
            model.summary_writer.add_scalar('loss/recon_loss', recon_loss, global_step=step)
            model.summary_writer.add_scalar('loss/loss', loss, global_step=step)
            model.summary_writer.add_scalar('metric/W_dist', W_dist, global_step=step)
            model.summary_writer.add_scalar('metric/gradient_penalty', gp, global_step=step)
            model.summary_writer.flush()

            if step % 20 == 0 and step != 0:
                fake = fake.detach().numpy().transpose(0, 3, 2, 1)
                fake = fake[:36]
                real = images.detach().numpy().transpose(0, 3, 2, 1)
                real = real[:36]
                save_fake_img = merge(fake, [6, 6])
                save_real_img = merge(real, [6, 6])
                img_fake_path = os.path.join(model.eval_path, 'step_{}_fake.png'.format(step))
                img_real_path = os.path.join(model.eval_path, 'step_{}_real.png'.format(step))
                scipy.misc.imsave(img_fake_path, save_fake_img)
                scipy.misc.imsave(img_real_path, save_real_img)

                auc = evaluation(model.ae, eval_loader)
                model.summary_writer.add_scalar('metric/auc', auc, global_step=step)
                aucs.append(auc)
                print(aucs)

                if auc > best_auc:
                    best_auc = auc
                    model_save_path = os.path.join(model.checkpoint_path, 'model.pth'.format(step))
                    torch.save(model.ae, model_save_path)
                    log('Best auc: {}'.format(best_auc))
                    log('Model of step {} has been save to {}'.format(step, model_save_path), level=3)
                else:
                    print('auc: {} best_auc:{}'.format(auc, best_auc))


if __name__ == '__main__':
    torch.manual_seed (2023)
    train()
    
 

    
