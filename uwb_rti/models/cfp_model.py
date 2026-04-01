import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class CFPModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Shortcut path: Conv2d(5x5, 1 filter)
        self.shortcut = nn.Conv2d(1, 1, kernel_size=5, padding=2)

        # Residual path
        self.res_entry = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
        )
        self.res_bn_relu = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # Post-merge layers
        self.merge_bn_relu = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv_48 = nn.Conv2d(32, 48, kernel_size=1)
        self.skip_proj = nn.Conv2d(32, 48, kernel_size=1)
        self.dropout = nn.Dropout(0.3)
        self.conv_out = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x):
        # Shortcut path
        shortcut = self.shortcut(x)  # (B, 1, 30, 30)

        # Residual path
        res = self.res_entry(x)  # (B, 32, 30, 30)
        res = self.res_blocks(res)  # (B, 32, 30, 30)
        res = self.res_bn_relu(res)  # (B, 32, 30, 30)

        # Merge: shortcut (1 ch) broadcast-added to residual (32 ch).
        # The single shortcut feature is added identically to all 32 channels.
        merge = shortcut + res  # (B, 32, 30, 30)

        # Post-merge with residual skip around Conv2d(1x1, 48).
        # skip_proj projects merge (32 ch) to 48 ch to match conv_48 output.
        out = self.merge_bn_relu(merge)  # (B, 32, 30, 30)
        out = self.conv_48(out) + self.skip_proj(merge)  # (B, 48, 30, 30)
        out = self.dropout(out)
        out = self.conv_out(out)  # (B, 1, 30, 30)

        return out
