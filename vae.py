import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, in_ch, img_size, z_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,32,4,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,4,2,1),   nn.ReLU(True),
            nn.Conv2d(64,128,4,2,1),  nn.ReLU(True),
            nn.Conv2d(128,256,4,2,1), nn.ReLU(True),
        )
        # discover shape
        with torch.no_grad():
            d = torch.zeros(1, in_ch, img_size, img_size)
            out = self.conv(d)
        self.C, self.H, self.W = out.shape[1:]
        self.flatten_dim = self.C * self.H * self.W

        self.fc_mu     = nn.Linear(self.flatten_dim, z_dim)
        # Predict log-variance of latent q(z | x), not decoder pixel variance.
        self.fc_logvar = nn.Linear(self.flatten_dim, z_dim)

    def forward(self, x):
        batch = x.size(0)
        h     = self.conv(x).view(batch, -1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, out_ch, img_size, z_dim):
        super().__init__()
        enc = Encoder(out_ch, img_size, z_dim)  # just for shape
        self.C, self.H, self.W = enc.C, enc.H, enc.W

        # This decoder predicts only the pixel means f_theta(z), with no
        # learned pixel-variance head. The fixed observation variance is implicit
        # in the Gaussian interpretation of the MSE loss in train_vae.py.
        self.fc = nn.Linear(z_dim, self.C*self.H*self.W)
        self.deconv = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(self.C,128,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,4,2,1),   nn.ReLU(True),
            nn.ConvTranspose2d(64, 32,4,2,1),    nn.ReLU(True),
            nn.ConvTranspose2d(32, out_ch,4,2,1),
            # Match training pixels: ToTensor maps [0,255] to [0,1], then
            # Normalize(mean=0.5, std=0.5) maps them to [-1,1] in utils.py.
            # Tanh bounds the predicted means; it does not set their variance.
            nn.Tanh(),
        )

    def forward(self, z):
        # z has shape [batch, z_dim]; current training uses z_dim=512.
        batch = z.size(0)
        # Project each latent vector to C*H*W values, then reshape to feature maps.
        # For 64x64 inputs: [batch, 512] -> [batch, 4096] -> [batch, 256, 4, 4].
        h = self.fc(z).view(batch, self.C, self.H, self.W)
        # Four stride-2 transposed convolutions grow spatial sizes 4->8->16->32->64.
        # Return the mean reconstruction [batch, 3, 64, 64] for RGB 64x64 images.
        return self.deconv(h)


class VAE(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 image_size:  int = 64,
                 z_dim:       int = 128):
        super().__init__()
        # pass in_channels & image_size down
        self.enc = Encoder(in_channels, image_size, z_dim)
        self.dec = Decoder(in_channels, image_size, z_dim)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.enc(x)
        z          = self.reparameterize(mu, logvar)
        recon      = self.dec(z)
        return recon, mu, logvar