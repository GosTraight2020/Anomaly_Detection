import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn import Sequential
from torch.utils.tensorboard import SummaryWriter
from torch import optim, autograd
import os 
from utils import log
def Conv2D(in_channels, out_channels, kernel_size, stride, padding, activation='lrelu'):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    torch.nn.init.xavier_uniform_(conv.weight)
    if activation == 'lrelu':
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    elif activation == 'relu':
        return nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Res_Block_up(nn.Module):
    def __init__(self, nf_input, nf_output, kernel_size=3):
        super(Res_Block_up, self).__init__()
        self.shortcut = Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_output, kernel_size=1, padding='same')
        )
        self.network = Sequential(
            nn.BatchNorm2d(num_features=nf_input, eps=1e-5, momentum=0.99),
            nn.ReLU(),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_output, kernel_size=kernel_size, padding='same',
                      bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=nf_output, eps=1e-5, momentum=0.99),
            nn.ReLU(),
            nn.Conv2d(in_channels=nf_output, out_channels=nf_output, kernel_size=kernel_size, padding='same'),
            nn.ReLU()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        residual = self.network(x)
        out = identity + residual
        return out

class Res_Block_down(nn.Module):
    def __init__(self, size, nf_input, nf_output, kernel_size=3):
        super(Res_Block_down, self).__init__()
        self.shortcut = Sequential(
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_output, kernel_size=1, padding='same'),
            nn.ReLU(),
        )
        self.network = Sequential(
            nn.LayerNorm(normalized_shape=[nf_input, size, size]),
            nn.ReLU(),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_input, kernel_size=kernel_size, padding='same', bias=False),
            nn.ReLU(),
            nn.LayerNorm(normalized_shape=[nf_input, size, size]),
            nn.ReLU(),
            nn.Conv2d(in_channels=nf_input, out_channels=nf_output, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2)
        )

    def forward(self, x):
        identity = self.shortcut(x)
        residual = self.network(x)
        out = identity + residual
        return out

class Critic_128(nn.Module):
    def __init__(self, num_channels, nf):
        super(Critic_128, self).__init__()
        self.nf = nf
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.nf * 1, kernel_size=3, padding='same')
        self.res_block1 = Res_Block_down(size=128, nf_input=self.nf * 1, nf_output=self.nf * 2)
        self.res_block2 = Res_Block_down(size=64, nf_input=self.nf * 2, nf_output=self.nf * 4)
        self.res_block3 = Res_Block_down(size=32, nf_input=self.nf * 4, nf_output=self.nf * 8)
        self.res_block4 = Res_Block_down(size=16, nf_input=self.nf * 8, nf_output=self.nf * 16)
        self.res_block5 = Res_Block_down(size=8, nf_input=self.nf * 16, nf_output=self.nf * 16)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=self.nf * 16 * 4 * 4, out_features=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class Encoder_128(nn.Module):
    def __init__(self, num_channels, nf):
        super(Encoder_128, self).__init__()
        self.nf = nf
        self.num_channels = num_channels
        self.conv1 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.nf * 1, kernel_size=3, padding='same')
        self.res_block1 = Res_Block_down(size=128, nf_input=self.nf * 1, nf_output=self.nf * 2)
        self.res_block2 = Res_Block_down(size=64, nf_input=self.nf * 2, nf_output=self.nf * 4)
        self.res_block3 = Res_Block_down(size=32, nf_input=self.nf * 4, nf_output=self.nf * 8)
        self.res_block4 = Res_Block_down(size=16, nf_input=self.nf * 8, nf_output=self.nf * 16)
        self.res_block5 = Res_Block_down(size=8, nf_input=self.nf * 16, nf_output=self.nf * 16)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=self.nf * 16 * 4 * 4, out_features=100)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

class Generator_128(nn.Module):
    def __init__(self, num_channels, nf):
        super(Generator_128, self).__init__()
        self.nf = nf
        self.num_channels = num_channels
        self.linear = nn.Linear(in_features=100, out_features=4*4*16*self.nf)
        self.res_block1 = Res_Block_up(self.nf * 16, self.nf * 16)
        self.res_block2 = Res_Block_up(self.nf * 16, self.nf * 8)
        self.res_block3 = Res_Block_up(self.nf * 8, self.nf * 4)
        self.res_block4 = Res_Block_up(self.nf * 4, self.nf * 2)
        self.res_block5 = Res_Block_up(self.nf * 2, self.nf * 1)
        self.bn = nn.BatchNorm2d(num_features=self.nf*1, eps=1e-5, momentum=0.99)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=self.nf*1, out_channels=self.num_channels, kernel_size=3, padding='same')


    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 16 * self.nf, 4, 4)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.conv1(x)
        return x


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
            Conv2D(in_channels=int(self.nf*0.5), out_channels=int(self.nf*0.5), kernel_size=3, stride=1, padding=1)
        )

        self.CBR5 = nn.Conv2d(in_channels=int(self.nf*0.5), out_channels=self.num_channels, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        # x = self.linear(x)
        # x = x.view(-1, 16 * self.nf, 4, 4)
        x = self.CBR1(x)
        x = self.CBR2(x)
        x = self.CBR3(x)
        x = self.CBR4(x)
        x = self.CBR5(x)
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
        # self.activation = nn.Sigmoid()

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
    def __init__(self, data_shape, in_channels, out_channels, nf):
        super(AE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nf = nf
        self.data_shape = data_shape

        # self.encoder = Encoder_128(self.in_channels, self.nf)
        self.encoder = DC_Encoder_128(self.in_channels, self.nf)
        self.decoder = DC_Generator_128(self.out_channels, self.nf)
        # self.decoder = Generator_128(self.out_channels, self.nf)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoEncoder:
    def __init__(self, dataset, data_shape, C_lr=1e-4, G_lr=1e-4, summary_path='./summary/', checkpoint_path='./checkpoints', ae_in_channels=3, ae_out_channels=3):
        self.C_lr = C_lr
        self.G_lr = G_lr
        self.lamda = 10.
        self.dataset = dataset
        self.data_shape = data_shape
        # self.critic = Critic_128(num_channels=ae_in_channels, nf=64)
        self.critic = DC_Critic_128(num_channels=ae_in_channels, nf=64)
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