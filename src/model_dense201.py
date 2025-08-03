import torch
import torch.nn as nn
from torchvision import models
from src.fusion_modules import MetadataEncoder, MultiScaleHierarchicalFusion

class DenseNetMultiScale(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        features = models.densenet201(pretrained=pretrained).features
        self.stage1 = nn.Sequential(
            features.conv0, features.norm0, features.relu0, features.pool0,
            features.denseblock1, features.transition1
        )
        self.stage2 = nn.Sequential(
            features.denseblock2, features.transition2
        )
        self.stage3 = nn.Sequential(
            features.denseblock3, features.transition3
        )
        self.stage4 = features.denseblock4

    def forward_scales(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        return [x2, x3, x4]

class MultiscaleFusionClassifier(nn.Module):
    def __init__(self, num_classes, meta_in, meta_out, K_init, ref_delta, ref_epochs, input_size=(224,224)):
        super().__init__()
        self.backbone = DenseNetMultiScale(pretrained=True)
        dummy = torch.randn(1, 3, *input_size)
        with torch.no_grad():
            fm = self.backbone.forward_scales(dummy)
        Ns = [f.shape[2] * f.shape[3] for f in fm]
        pooling_ratios = [k/n for k, n in zip(K_init, Ns)]
        self.proj_img = nn.ModuleList([nn.Linear(f.shape[1], meta_out) for f in fm])
        self.meta_enc = MetadataEncoder(meta_in, meta_out)
        self.fusion   = MultiScaleHierarchicalFusion(meta_out, meta_out, pooling_ratios, meta_out, heads=1, layers=1)
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
