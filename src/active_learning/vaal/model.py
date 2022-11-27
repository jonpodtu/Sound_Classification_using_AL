import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

"""
The purpot of this document is to build up the 2 out of 3 neural networks that are used in the VAAL architecture:
    - A Variational Auto Encoder (VAE)
    - A Discriminator

The 3rd model is the task learner which we in the main script import seperately as this will vary between experiments
"""

# All below is taken and slightly changed from VAAL GitHub: https://github.com/sinhasam/vaal
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class PrintSize(nn.Module):
    """Utility module to print current shape of a Tensor in Sequential, only at the first pass."""

    first = True

    def forward(self, x):
        if self.first:
            print(f"Size: {x.size()}")
            self.first = False
        return x


class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""

    def __init__(self, z_dim=32, nc=1, multiplier=2):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=nc,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),  # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * (multiplier * 2) * (multiplier * 2))),  # B, 1024*4*4
        )

        # CHANGE BELOW!
        self.fc_mu = nn.Linear(
            1024 * (multiplier * 2) * (multiplier * 2), z_dim
        )  # B, z_dim
        self.fc_logvar = nn.Linear(
            1024 * (multiplier * 2) * (multiplier * 2), z_dim
        )  # B, z_dim

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * (multiplier * 4) * (multiplier * 4)),  # B, 1024*4*4
            View((-1, 1024, (multiplier * 4), (multiplier * 4))),  # B, 1024,  4,  3
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # B,  512, 8, 18
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),  # B, 1, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        x = x.unsqueeze(1)  # THIS SHOULD BE CHECKED FOR THE SWEET LOVE OF CHRIST
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z).squeeze(1)

    def PrintParameters(self):
        for child in self.encoder.children():
            print(child)
        for param in list(child.parameters()):
            print(param[0])


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""

    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, z):
        return self.net(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

