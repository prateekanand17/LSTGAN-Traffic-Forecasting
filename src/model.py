import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import *

class TemporalEncoder(nn.Module):
    def __init__(self, in_ch, out_ch, s1, s2):
        super().__init__()
        mid = max((in_ch + out_ch) // 2, 8)
        self.c1a = nn.Conv1d(in_ch, mid, 3, padding=1)
        self.c1b = nn.Conv1d(mid, mid, s1, stride=s1) if s1 > 1 else nn.Identity()
        self.bn1 = nn.BatchNorm1d(mid)
        self.c2a = nn.Conv1d(mid, out_ch, 3, padding=1)
        self.c2b = nn.Conv1d(out_ch, out_ch, s2, stride=s2) if s2 > 1 else nn.Identity()
        self.bn2 = nn.BatchNorm1d(out_ch)
    def forward(self, x):
        B, T, N, C = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        x = F.relu(self.bn1(self.c1b(F.relu(self.c1a(x)))))
        x = self.bn2(self.c2b(F.relu(self.c2a(x))))
        _, Co, To = x.shape
        return x.reshape(B, N, Co, To).permute(0, 3, 1, 2)

class SpatialEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(HOURLY_IN_CHANNELS, SPATIAL_EMBED_DIM, 1)
        self.bn = nn.BatchNorm1d(SPATIAL_EMBED_DIM)
    def forward(self, x):
        B, T, N, C = x.shape
        x = x.reshape(B * T, N, C).permute(0, 2, 1)
        return self.bn(self.conv(x)).permute(0, 2, 1).reshape(B, T, N, -1)

class GlobalSpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({
            'attn': nn.MultiheadAttention(SPATIAL_EMBED_DIM, GLOBAL_ATTN_HEADS, batch_first=True, dropout=0.1),
            'n1': nn.LayerNorm(SPATIAL_EMBED_DIM), 'n2': nn.LayerNorm(SPATIAL_EMBED_DIM),
            'ffn': nn.Sequential(nn.Linear(SPATIAL_EMBED_DIM, SPATIAL_EMBED_DIM * 4),
                                 nn.GELU(), nn.Dropout(0.1), nn.Linear(SPATIAL_EMBED_DIM * 4, SPATIAL_EMBED_DIM))
        })])
        self.proj = nn.Identity()
    def forward(self, x):
        B, T, N, C = x.shape; x = x.reshape(B * T, N, C)
        for l in self.layers:
            a, _ = l['attn'](x, x, x); x = l['n1'](x + a); x = l['n2'](x + l['ffn'](x))
        return self.proj(x).reshape(B, T, N, -1)

class ManualChebConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(torch.empty(SPATIAL_EMBED_DIM, LOCAL_SPATIAL_CHANNELS)) for _ in range(CHEBYSHEV_K)
        ])
        self.bias = nn.Parameter(torch.zeros(LOCAL_SPATIAL_CHANNELS))
        for w in self.weights: nn.init.xavier_uniform_(w)
    def forward(self, x, L):
        Z0, Z1 = x, L @ x; out = Z0 @ self.weights[0]
        if CHEBYSHEV_K > 1: out = out + Z1 @ self.weights[1]
        for k in range(2, CHEBYSHEV_K):
            Zk = 2 * L @ Z1 - Z0; out = out + Zk @ self.weights[k]; Z0, Z1 = Z1, Zk
        return out + self.bias

class LocalSpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cheb = ManualChebConv(); self.register_buffer('L_scaled', torch.zeros(NUM_SENSORS, NUM_SENSORS))
    def set_graph(self, adj_mx):
        import numpy as np
        A = adj_mx; D_inv = np.diag(1.0 / np.sqrt(np.maximum(A.sum(1), 1e-8)))
        L = np.eye(A.shape[0]) - D_inv @ A @ D_inv; eig = np.linalg.eigvalsh(L)
        self.L_scaled.copy_(torch.FloatTensor(2 * L / (eig[-1] + 1e-8) - np.eye(A.shape[0])))
    def forward(self, x, edge_idx=None, edge_wt=None):
        B, T, N, C = x.shape; L = self.L_scaled; x = x.reshape(B * T, N, C)
        return self.cheb(x, L).reshape(B, T, N, -1)

class TimeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        h = D_MODEL * 2
        self.mlp = nn.Sequential(nn.Linear(TIME_ONEHOT_DIM, h), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, D_MODEL))
        self.norm = nn.LayerNorm(D_MODEL)
    def forward(self, ste, ti):
        dow = F.one_hot(ti[:, 0].long(), 7).float()
        tod = F.one_hot(ti[:, 1].long(), 288).float()
        vt = self.mlp(torch.cat([dow, tod], -1))[:, None, None, :].expand_as(ste)
        return self.norm(ste + vt)

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        pe = torch.zeros(500, D_MODEL); pos = torch.arange(500).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D_MODEL, 2).float() * (-math.log(10000.0) / D_MODEL))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TemporalDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pe = PositionalEncoding()
        self.enc_layers = nn.ModuleList([nn.ModuleDict({
            'attn': nn.MultiheadAttention(D_MODEL, DECODER_HEADS, batch_first=True, dropout=0.1),
            'n1': nn.LayerNorm(D_MODEL), 'n2': nn.LayerNorm(D_MODEL),
            'ffn': nn.Sequential(nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(D_MODEL * 4, D_MODEL))
        })])
        self.dec_layers = nn.ModuleList([nn.ModuleDict({
            'attn': nn.MultiheadAttention(D_MODEL, DECODER_HEADS, batch_first=True, dropout=0.1),
            'n1': nn.LayerNorm(D_MODEL), 'n2': nn.LayerNorm(D_MODEL),
            'ffn': nn.Sequential(nn.Linear(D_MODEL, D_MODEL * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(D_MODEL * 4, D_MODEL))
        })])
        self.fq = nn.Parameter(torch.randn(1, FORECAST_HORIZON, D_MODEL) * 0.02)
        self.out_proj = nn.Linear(D_MODEL, 1)
    def forward(self, ste):
        B, T, N, d = ste.shape; x = ste.permute(0, 2, 1, 3).reshape(B * N, T, d); x = self.pe(x)
        for l in self.enc_layers:
            a, _ = l['attn'](x, x, x); x = l['n1'](x + a); x = l['n2'](x + l['ffn'](x))
        q = self.fq.expand(B * N, -1, -1)
        for l in self.dec_layers:
            a, _ = l['attn'](q, x, x); q = l['n1'](q + a); q = l['n2'](q + l['ffn'](q))
        return self.out_proj(q).reshape(B, N, FORECAST_HORIZON, 1).permute(0, 2, 1, 3)

class LSTGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.weekly_enc = TemporalEncoder(WEEKLY_IN_CHANNELS, WEEKLY_OUT_CHANNELS, WEEKLY_STRIDE_1, WEEKLY_STRIDE_2)
        self.daily_enc  = TemporalEncoder(DAILY_IN_CHANNELS, DAILY_OUT_CHANNELS, DAILY_STRIDE_1, DAILY_STRIDE_2)
        self.sp_embed   = SpatialEmbedding()
        self.global_enc = GlobalSpatialEncoder()
        self.local_enc  = LocalSpatialEncoder()
        self.time_enc   = TimeEncoder()
        self.decoder    = TemporalDecoder()
    def _match_time(self, x, tgt):
        T = x.shape[1]
        if T == tgt: return x
        if T > tgt: return x[:, :tgt]
        B, T, N, C = x.shape; x = x.permute(0, 3, 2, 1).reshape(B * C * N, 1, T)
        x = F.interpolate(x, size=tgt, mode='nearest'); return x.reshape(B, C, N, tgt).permute(0, 3, 2, 1)
    def forward(self, X_w, X_d, X_h, t_info):
        Xw = self.weekly_enc(X_w); Xd = self.daily_enc(X_d); Xe = self.sp_embed(X_h)
        Xsa = self.global_enc(Xe); Xgcn = self.local_enc(Xe)
        Tt = X_h.shape[1]; Xw = self._match_time(Xw, Tt); Xd = self._match_time(Xd, Tt)
        STe = torch.cat([Xw, Xd, Xsa, Xgcn], dim=-1); STE = self.time_enc(STe, t_info)
        return self.decoder(STE)
