import torch
import torch.nn as nn
from torchvision import models
from src.fusion_modules import MetadataEncoder, MultiScaleHierarchicalFusion

class ConvNeXtTinyMultiScale(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.convnext_tiny(pretrained=pretrained)
        self.features = base.features

    def forward_scales(self, x):
        f = self.features
        x = f[0](x)
        x = f[1](x)
        x = f[2](x)
        x = f[3](x)
        x2 = x
        x = f[4](x)
        x = f[5](x)
        x3 = x
        x = f[6](x)
        x = f[7](x)
        x4 = x
        return [x2, x3, x4]

class MultiscaleFusionClassifier(nn.Module):
    def __init__(self, num_classes, meta_in, meta_out, K_init, ref_delta, ref_epochs, input_size=(224,224)):
        super().__init__()
        self.backbone = ConvNeXtTinyMultiScale(pretrained=True)
        dummy = torch.randn(1, 3, *input_size)
        with torch.no_grad():
            fm = self.backbone.forward_scales(dummy)
        Ns = [f.shape[2] * f.shape[3] for f in fm]
        pooling_ratios = [k / n for k, n in zip(K_init, Ns)]
        self.proj_img = nn.ModuleList([nn.Linear(f.shape[1], meta_out) for f in fm])
        self.meta_enc = MetadataEncoder(meta_in, meta_out)
        self.fusion   = MultiScaleHierarchicalFusion(meta_out, meta_out, pooling_ratios, meta_out)
        self.head     = nn.Linear(meta_out, num_classes)

    def forward(self, x, meta):
        fmaps = self.backbone.forward_scales(x)
        nodes = []
        for fmap, proj in zip(fmaps, self.proj_img):
            B, C, H, W = fmap.shape
            flat = fmap.view(B, C, H*W).transpose(1, 2)
            nodes.append(proj(flat))
        mfeat = self.meta_enc(meta)
        fused = self.fusion(nodes, mfeat)
        return self.head(fused)
