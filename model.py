import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # -> [16, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [16, H/2, W/2]
            nn.Conv2d(16, 32, kernel_size=3, padding=1), # -> [32, H/2, W/2]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # -> [32, H, W]
            nn.Conv2d(32, 16, kernel_size=3, padding=1),                         # -> [16, H, W]
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1)                                     # -> [1, H, W]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
