import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MetadataEncoder(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * in_size, out_size), nn.ReLU()
        )

    def forward(self, x):
        return self.net(x.unsqueeze(1))

class EdgeUncertaintyGATLayer(nn.Module):
    def __init__(self, in_f, out_f, meta_dim, hidden_edge=64, dropout=0.0, alpha=0.2, prune_ratio=0.1):
        super().__init__()
        self.W = nn.Linear(in_f, out_f, bias=False)
        self.drop = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_f + meta_dim, hidden_edge), nn.ReLU(),
            nn.Linear(hidden_edge, 1)
        )
        self.prune_ratio = prune_ratio
        self.unc = nn.Sequential(
            nn.Linear(in_f, in_f // 2), nn.ReLU(),
            nn.Linear(in_f // 2, 1), nn.Sigmoid()
        )

    def forward(self, h, meta):
        Wh = self.W(h)
        B, N, _ = Wh.shape
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        mexp = meta.unsqueeze(1).unsqueeze(1).expand(B, N, N, -1)
        eij = self.edge_mlp(torch.cat([Wh_i, Wh_j, mexp], -1)).squeeze(-1)
        k = max(1, math.ceil(self.prune_ratio * N))
        _, indices = torch.topk(eij, k, dim=-1)
        mask = torch.zeros_like(eij)
        batch_idx = torch.arange(B).view(B,1,1)
        node_i_idx = torch.arange(N).view(1,N,1)
        mask[batch_idx, node_i_idx, indices] = 1.0
        e = eij * mask
        att = F.softmax(e, dim=-1)
        att = self.drop(att)
        u = self.unc(h).squeeze(-1)
        att = att * u.unsqueeze(1)
        return torch.matmul(att, Wh)

class MultiHeadEUGAT(nn.Module):
    def __init__(self, in_f, out_f, meta_dim, heads=4, **kw):
        super().__init__()
        self.heads = nn.ModuleList([
            EdgeUncertaintyGATLayer(in_f, out_f, meta_dim, **kw)
            for _ in range(heads)
        ])
        self.fc = nn.Linear(heads * out_f, out_f)

    def forward(self, h, meta):
        return self.fc(torch.cat([hd(h, meta) for hd in self.heads], -1))

class SelfAttentionGraphPooling(nn.Module):
    def __init__(self, in_f, ratio, hidden=None):
        super().__init__()
        self.ratio = ratio
        self.h = hidden or in_f
        self.fc1 = nn.Linear(in_f, self.h)
        self.fc2 = nn.Linear(self.h, 1)

    def forward(self, h):
        B, N, _ = h.shape
        scores = self.fc2(torch.tanh(self.fc1(h))).squeeze(-1)
        K = max(1, math.ceil(self.ratio * N))
        topk_inds = torch.topk(scores, K, dim=1).indices
        A = torch.zeros(B, K, N, device=h.device)
        for b in range(B):
            A[b, torch.arange(K), topk_inds[b]] = 1.0
        return torch.bmm(A, h)

class MultiScaleHierarchicalFusion(nn.Module):
    def __init__(self, in_f, out_f, pooling_ratios, meta_dim, heads=4, layers=2, dropout=0.1, alpha=0.2, prune_ratio=0.1):
        super().__init__()
        self.poolings = nn.ModuleList([
            SelfAttentionGraphPooling(in_f, r) for r in pooling_ratios
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_f))
        self.layers = nn.ModuleList([
            MultiHeadEUGAT(in_f, in_f, meta_dim,
                           heads=heads, dropout=dropout,
                           alpha=alpha, prune_ratio=prune_ratio)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(in_f)
        self.proj = nn.Linear(in_f, out_f)

    def forward(self, hs, meta):
        B = meta.size(0)
        pooled = [p(h) for p, h in zip(self.poolings, hs)]
        mnode = meta.unsqueeze(1)
        gnode = self.cls_token.expand(B, -1, -1)
        h = torch.cat(pooled + [mnode, gnode], 1)
        for lyr in self.layers:
            res = h
            h = torch.nn.functional.elu(lyr(h, meta))
            h = self.norm(h + res)
        return self.proj(h[:, -1])
