# basicos
import numpy as np
import math 

# redes neuronales
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft 

from dataclasses import dataclass
@dataclass
class Configs:
    # Minimal defaults — adjust these values to match your needs
    n_features: int = 1
    n_targets: int = 1
    pred_len: int = 1
    top_k: int = 2
    dropout: float = 0.0
    d_model: int = 32
    d_ff: int = 64
    num_kernels: int = 6
    num_times_blocks: int = 1
    hidden: int = 32
    layers: int = 1
    bidirectional: bool = False

def FFT_for_Period(x, k=2):
    """
    x: [B, T, C] float tensor (in d_model)
    Outputs:
      period_list: np.array of shape [k] with periods (int >=1)
      period_weight: [B, k] average amplitudes per sample in the top k frequencies
    """
    # FFT in temporal dimension
    xf = torch.fft.rfft(x, dim=1)  # [B, T//2+1, C] in complex magnitude
    amp = torch.abs(xf)            # [B, F, C]

    # Average over batch and channels for global frequency ranking
    A = amp.mean(dim=0).mean(dim=-1)  # [F]
    if A.shape[0] > 0:
        A[0] = 0.0  # ignore zero frequency

    # top-k frequencies
    k_eff = min(k, A.shape[0])
    _, top_idx = torch.topk(A, k_eff)
    top_idx = top_idx.detach().cpu().numpy()

    # Ensure indices >=1 to avoid division by zero
    top_idx = np.clip(top_idx, 1, None)

    # Periods = ceil(T / f)
    T = x.shape[1]
    periods = np.ceil(T / top_idx).astype(int)
    periods = np.clip(periods, 1, None)  # avoid 0

    # Weights per sample: mean over channels and gather at top_idx
    per_sample_amp = amp.mean(dim=-1)    # [B, F]
    # Adjustment if k_eff < k
    if k_eff < k:
        # pad with the last valid frequency to maintain shape
        pad_count = k - k_eff
        pad_idx = np.full((pad_count,), top_idx[-1], dtype=top_idx.dtype)
        gather_idx = np.concatenate([top_idx, pad_idx], axis=0)
    else:
        gather_idx = top_idx
    gather_idx_t = torch.tensor(gather_idx, device=x.device, dtype=torch.long)
    period_weight = per_sample_amp[:, gather_idx_t]  # [B, k]

    return periods, period_weight

##### TimesNet #####

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2*i+1, padding=i)
            for i in range(num_kernels)
        ])
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        if init_weight:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # x: [B, C_in, H, W]
        outs = [k(x) for k in self.kernels]  # each [B, C_out, H, W]
        # Stack and mean over the last new dim
        x = torch.stack(outs, dim=-1).mean(dim=-1)  # [B, C_out, H, W]
        return x

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        self.dropout = nn.Dropout(configs.dropout)
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x, extend_len=None):
        # x: [B, T, C] (C = d_model)
        B, T, C = x.size()
        if extend_len is None:
            extend_len = self.pred_len

        periods, period_weight = FFT_for_Period(x, self.k)  # periods: np[k], period_weight: [B, k]

        res = []
        target_T = T + max(extend_len, 0)
        for i in range(self.k):
            p = int(periods[i])
            p = max(p, 1)
            target_len = math.ceil(target_T / p) * p
            pad_len = target_len - T
            if pad_len > 0:
                padding = x.new_zeros(B, pad_len, C)
                out = torch.cat([x, padding], dim=1)    # [B, target_len, C]
            else:
                out = x                                  # [B, T, C]

            # 1D -> 2D
            out = out.reshape(B, target_len // p, p, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            out = self.conv(out)           # [B, C, H, W]
            # 2D -> 1D
            out = out.permute(0, 2, 3, 1).reshape(B, -1, C)[:, :target_T, :]  # [B, target_T, C]
            res.append(out)

        res = torch.stack(res, dim=-1)  # [B, target_T, C, k]

        # Weighting by amplitude (softmax over k) and replication
        w = F.softmax(period_weight, dim=1).unsqueeze(1).unsqueeze(1)  # [B,1,1,k]
        w = w.repeat(1, target_T, C, 1)                                # [B,target_T,C,k]
        res = torch.sum(res * w, dim=-1)                               # [B, target_T, C]

        # Residual with padding to the future
        if target_T > T:
            x_pad = torch.cat([x, x.new_zeros(B, target_T - T, C)], dim=1)
        else:
            x_pad = x[:, :target_T, :]
        res = res + x_pad
        return res  # [B, target_T, C]

class TimesNet(nn.Module):
    def __init__(self, configs: Configs):
        super().__init__()
        n_features_in = configs.n_features
        n_targets = configs.n_targets
        self.pred_len = configs.pred_len
        self.blocks = nn.ModuleList([TimesBlock(configs) for _ in range(configs.num_times_blocks)])
        self.norm = nn.LayerNorm(configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        # Embedding: N -> d_model
        self.input_proj = nn.Linear(n_features_in, configs.d_model)
        # Projecting output: d_model -> n_targets
        self.out_proj = nn.Linear(configs.d_model, n_targets)

    def forward_features(self, x):
        # x: [B, T, N_in]
        x = self.norm(self.input_proj(x))              # [B, T, d_model] = [B, T, C]
        for i, block in enumerate(self.blocks):
            extend = self.pred_len if i == 0 else 0
            x = block(x, extend_len=extend)            # extend 1st block
            x = self.dropout(x)
        # keep the last timestep as embedding
        feats_last = x[:, -1, :]                       # [B, d_model]
        return feats_last
        
    def forward(self, x):
        feats_last = self.forward_features(x)          # [B, d_model]
        y_hat = self.out_proj(feats_last).view(x.size(0), -1)  # [B, n_targets]
        return y_hat

##### BiLSTM #####

class BiLSTM(nn.Module):
    def __init__(self, configs: Configs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=configs.n_features, 
            hidden_size=configs.hidden, 
            num_layers=configs.layers,
            batch_first=True,      # (B, L, input_size)
            bidirectional=configs.bidirectional,
            dropout=configs.dropout
        )

    def forward(self, x):
        # out -> [B, L, hidden * num_directions]
        out, _ = self.lstm(x)
        return out

##### Parallel model #####
        
class TimesNet_BiLSTM_Parallel(nn.Module):
    def __init__(self, configs: Configs, device="cuda"):
        
        super().__init__()
        self.times = TimesNet(configs=configs)
        self.bilstm = BiLSTM(configs=configs)

        # Feature dimensions for the FC:
        d_model = configs.d_model
        lstm_hidden = configs.hidden
        n_targets = configs.n_targets
        if configs.bidirectional:
            self.head = nn.Sequential(
                # nn.Linear(d_model + lstm_hidden * 2, 64),
                # nn.ReLU(),
                # nn.Dropout(0.2),
                # nn.Linear(64, n_targets),
                nn.Linear(d_model + lstm_hidden * 2, n_targets) # [32+32*2,1]
            )
        else:
            self.head = nn.Sequential(
                # nn.Linear(d_model + lstm_hidden, 64),
                # nn.ReLU(),
                # nn.Dropout(0.2),
                nn.Linear(d_model + lstm_hidden, n_targets), # [32+32,1]
            )            

    def forward(self, x):
        # both branches receive the SAME x: [B, L, N_in]
        # print(f"x shape en Parallel model: {x.shape}")
        out1_last = self.times.forward_features(x)   # [B, d_model]
        out2_seq  = self.bilstm(x)                   # [B, L, H*D]
        out2_last = out2_seq[:, -1, :]               # [B, H*D]
        x_concat  = torch.cat([out1_last, out2_last], dim=-1)  # [B, d_model+H*D]
        y = self.head(x_concat).view(x.size(0), -1)  # [B, n_targets]
        return y