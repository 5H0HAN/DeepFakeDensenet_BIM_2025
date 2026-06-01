import torch
import torch.nn as nn
from torchvision import models

# CBAM Block
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa
        return x

# SE Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.se(x)
        return x * w

# Hybrid Ensemble Model
class HEACNN(nn.Module):
    def __init__(self, freeze_backbones=True, dropout=0.5, pretrained=True):
        super().__init__()

        weights_rn = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet50(weights=weights_rn)
        if freeze_backbones:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet_cbam = CBAMBlock(2048)

        weights_eff = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.efficientnet = models.efficientnet_b0(weights=weights_eff)
        if freeze_backbones:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        weights_vgg = models.VGG16_Weights.DEFAULT if pretrained else None
        self.vgg = models.vgg16(weights=weights_vgg).features
        if freeze_backbones:
            for param in self.vgg.parameters():
                param.requires_grad = False
        self.vgg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.vgg_se = SEBlock(512)

        self.resnet_fc = nn.Linear(2048, 1)
        self.efficientnet_fc = nn.Linear(1000, 1)
        self.vgg_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 1)
        )

        self.fusion = nn.Sequential(
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        rn_feats = self.resnet.forward_features(x) if hasattr(self.resnet, 'forward_features') else self.resnet.forward(x)
        if isinstance(rn_feats, torch.Tensor) and rn_feats.dim() == 2:
            rn_feats = rn_feats.unsqueeze(-1).unsqueeze(-1)
        rn_feats = self.resnet_cbam(rn_feats)
        rn_feats = torch.flatten(nn.AdaptiveAvgPool2d(1)(rn_feats), 1)
        rn_out = self.resnet_fc(rn_feats)

        eff_out = self.efficientnet(x)
        eff_out = self.efficientnet_fc(eff_out)

        vgg_feats = self.vgg(x)
        vgg_feats = self.vgg_se(vgg_feats)
        vgg_feats = self.vgg_pool(vgg_feats)
        vgg_out = self.vgg_fc(vgg_feats)

        logits = torch.cat([rn_out, eff_out, vgg_out], dim=1)
        output = self.fusion(logits)
        return output
