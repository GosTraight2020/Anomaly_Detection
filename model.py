import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
from torch import optim, autograd
import os 
from utils import log
def Conv2D(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    torch.nn.init.xavier_uniform_(conv.weight)
    return nn.Sequential(
        conv,
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(inplace=True)
    )

class DC_Generator_128(nn.Module):
    def __init__(self, num_channels, nf):
        super(DC_Generator_128, self).__init__()
        self.nf = nf
        self.num_channels = num_channels
        # self.linear = nn.Linear(in_features=100, out_features=4*4*16*self.nf)
        self.CBR1 = Sequential(
            nn.ConvTranspose2d(8*self.nf, 4*self.nf, kernel_size=2, stride=2, padding=0),
            Conv2D(in_channels=4*self.nf, out_channels=4*self.nf, kernel_size=3, stride=1, padding=1)
        )
        self.CBR2 = Sequential(
            nn.ConvTranspose2d(4*self.nf, 2*self.nf, kernel_size=2, stride=2, padding=0),
            Conv2D(in_channels=2*self.nf, out_channels=2*self.nf, kernel_size=3, stride=1, padding=1)
        )
        self.CBR3 = Sequential(
            nn.ConvTranspose2d(2*self.nf, 1*self.nf, kernel_size=2, stride=2, padding=0),
            Conv2D(in_channels=1*self.nf, out_channels=1*self.nf, kernel_size=3, stride=1, padding=1)
        )
        self.CBR4 = Sequential(
            nn.ConvTranspose2d(1*self.nf, int(self.nf*0.5), kernel_size=2, stride=2, padding=0),
            nn.Conv2d(in_channels=int(self.nf*0.5), out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # x = self.linear(x)
        # x = x.view(-1, 16 * self.nf, 4, 4)
        x = self.CBR1(x)
        x = self.CBR2(x)
        x = self.CBR3(x)
        x = self.CBR4(x)
        out = x
        return out

class DC_Encoder_128(nn.Module):
    def __init__(self, num_channels, nf):
        super(DC_Encoder_128, self).__init__()
        self.nf = nf
        self.num_channels = num_channels
        # self.CBR1 = Sequential(
        #     Conv2D(in_channels=self.num_channels, out_channels=self.nf*0.5, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )

        self.CBR1 = Sequential(
            Conv2D(in_channels=self.num_channels, out_channels=self.nf, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # self.CBR2 = Sequential(
        #     Conv2D(in_channels=self.nf*0.5, out_channels=self.nf, kernel_size=3, stride=1, padding=1),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        self.CBR3 = Sequential(
            Conv2D(in_channels=self.nf, out_channels=2*self.nf, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.CBR4 = Sequential(
            Conv2D(in_channels=2*self.nf, out_channels=4*self.nf, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.CBR5 = Sequential(
            Conv2D(in_channels=4*self.nf, out_channels=8*self.nf, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # self.flatten = nn.Flatten()
        # self.linear = nn.Linear(in_features=4*4*16*self.nf, out_features=100)

    def forward(self, x):
        x = self.CBR1(x)
        # x = self.CBR2(x)
        x = self.CBR3(x)
        x = self.CBR4(x)
        x = self.CBR5(x)
        # x = self.CBR5(x)
        # x = self.flatten(x)
        # out = self.linear(x)
        out = x
        return out

class DC_Critic_128(nn.Module):
    def __init__(self, num_channels, nf):
        super(DC_Critic_128, self).__init__()
        self.nf = nf
        self.num_channels = num_channels
        self.CBR1 = Sequential(
            Conv2D(in_channels=self.num_channels, out_channels=self.nf, kernel_size=3, stride=2, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.CBR2 = Sequential(
            Conv2D(in_channels=self.nf, out_channels=2*self.nf, kernel_size=3, stride=2, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.CBR3 = Sequential(
            Conv2D(in_channels=2*self.nf, out_channels=4*self.nf, kernel_size=3, stride=2, padding=1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.CBR4 = Sequential(
            Conv2D(in_channels=4*self.nf, out_channels=8*self.nf, kernel_size=3, stride=2, padding=1),
            
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=self.nf*8*8*8, out_features=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.CBR1(x)
        x = self.CBR2(x)
        x = self.CBR3(x)
        x = self.CBR4(x)
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.activation(x)
        out = x 
        return out

class AE(nn.Module):
    def __init__(self, data_shape, in_channels, out_channels, nf, g_activation='tanh'):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nf = nf
        self.data_shape = data_shape

        self.encoder = DC_Encoder_128(self.in_channels, self.nf)
        self.decoder = DC_Generator_128(self.out_channels, self.nf)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder:
    def __init__(self, dataset, data_shape, C_lr=1e-5, G_lr=1e-4, summary_path='./summary/', checkpoint_path='./checkpoints', ae_in_channels=3, ae_out_channels=3):
        self.C_lr = C_lr
        self.G_lr = G_lr
        self.lamda = 10.
        self.dataset = dataset
        self.data_shape = data_shape
        self.critic = DC_Critic_128(num_channels=ae_in_channels, nf=32)
        self.ae = AE(data_shape=self.data_shape, in_channels=ae_in_channels, out_channels=ae_out_channels, nf=64)
        self.C_opt = optim.Adam(self.critic.parameters(), lr=self.C_lr, betas=(0.0, 0.9))
        self.G_opt = optim.Adam(self.ae.parameters(), lr=self.G_lr, betas=(0.0, 0.9))
        self.l1_loss = nn.L1Loss()

        self.summary_path = os.path.join(summary_path, 'AE', dataset+'_'+str(self.data_shape[1]))
        self.checkpoint_path = os.path.join(checkpoint_path, 'AE', dataset+'_'+str(self.data_shape[1]))
        self.eval_path = self.checkpoint_path.replace('checkpoints', 'eval')
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path, exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)
        if not os.path.exists(self.eval_path):
            os.makedirs(self.eval_path, exist_ok=True)
        self.summary_writer = SummaryWriter(self.summary_path, flush_secs=30)
        log('Model has been created!')

    def calc_gradient_penalty(self, real_data, fake_data):
        batch_size, c, h, w = real_data.shape
        alpha = torch.rand([batch_size, 1, 1, 1]).repeat(1, c, h, w)
        interpolate = alpha * real_data + (1-alpha) * fake_data
        C_inter = self.critic(interpolate)

        grad = autograd.grad(
            outputs=C_inter, 
            inputs=interpolate, 
            grad_outputs=torch.ones_like(C_inter),
            create_graph=True, 
            retain_graph=True)[0]

        grad = grad.view(grad.size(0), -1)
        gp = ((grad.norm(2, dim=1)-1)**2).mean()
        return gp
    
    def train_critic_one_epoch(self, feat):
        fake = self.ae(feat)
        C_real = self.critic(feat)
        C_fake = self.critic(fake)
        gp = self.calc_gradient_penalty(feat, fake)
        W_dist = C_real.mean() - C_fake.mean()
        C_loss = -W_dist + self.lamda*gp
        self.critic.zero_grad()
        C_loss.backward(retain_graph=True)
        self.C_opt.step()
        return C_loss, W_dist, gp

    def train_generator_one_epoch(self, feat):
        fake = self.ae(feat)
        C_fake = self.critic(fake)
        G_loss = -torch.mean(C_fake)

        recon_loss = self.l1_loss(fake, feat)
        loss = G_loss + 50 * recon_loss
        self.ae.zero_grad()
        loss.backward()
        self.G_opt.step()
        return G_loss, recon_loss, loss, fake



if __name__ == '__main__':

    encoder = DC_Encoder_128(num_channels=3, nf=64)
    decoder = DC_Generator_128(num_channels=3, nf=64)
    summary_writer = SummaryWriter('./test', flush_secs=30)
    # summary_writer.add_graph(encoder, torch.randn(128, 3, 128, 128))
    summary_writer.add_graph(decoder, torch.randn(128, 512, 8, 8))
    summary_writer.close()