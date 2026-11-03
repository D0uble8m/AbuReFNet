import os
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from Loss import (
    abundance_sparsity_loss,
    endmember_anchor_loss,
    standard_normal_kl_loss,
)


UnmixingOutput = namedtuple(
    "UnmixingOutput",
    [
        "regional_endmembers",  # 每个patch对应的一组区域自适应端元[B, K, C]
        "abundance",    # 丰度 [B, K, H, W]， 每个像素对 K 个区域端元的丰度分布。
        "reconstruction",   # 通过端元和丰度重建的HSI [B, C, H, W]
        "similarity",       # 光谱相似度 [B, K, H, W]，每个像素对 K 个区域端元的光谱相似度。
        "delta_endmembers", # 区域适配器预测的端元残差 [B, K, C]。
        "residual_gate",
        "regional_context", # 融合空间上下文与平均光谱后的区域条件 [B, D]。
        "latent_mu",        # 单编码器近似后验的均值 [B, K, D_z]。
        "latent_logvar",    # 单编码器近似后验的对数方差 [B, K, D_z]。
        "latent",           # 重参数采样值或推理均值 [B, K, D_z]。
        "kl_loss",          # q(z|E_g,X) 与固定 N(0,I) 的 KL 散度。
        "anchor_loss",
        "abundance_sparse_loss",
    ],
)


def _default_vca_path():
    output_dir = globals().get("save_dir", ".")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    return os.path.join(output_dir, globals().get("vca_save_name", "vca_result.mat"))


def _format_endmembers(endmembers, num_bands=None):
    """Return endmembers as [K, B]. VCA_estimate.py saves them as [B, K]."""
    if not torch.is_tensor(endmembers):
        endmembers = torch.as_tensor(endmembers)
    endmembers = endmembers.float().squeeze()
    if endmembers.ndim != 2:
        raise ValueError(f"Expected a 2D endmember matrix, got {tuple(endmembers.shape)}.")

    if num_bands is not None:
        if endmembers.shape[1] == num_bands:
            return endmembers.contiguous()
        if endmembers.shape[0] == num_bands:
            return endmembers.t().contiguous()
        raise ValueError(
            f"Endmember bands do not match HSI bands: endmembers={tuple(endmembers.shape)}, "
            f"num_bands={num_bands}."
        )

    if endmembers.shape[0] > endmembers.shape[1]:
        return endmembers.t().contiguous()
    return endmembers.contiguous()

def load_vca_endmembers(vca_path=None, num_bands=None):
    """
    从.mat文件中读取估计的VCA端元,把端元矩阵格式化为[K, B],其中K是端元数量,B是光谱波段数量。
    """
    if vca_path is None:
        vca_path = _default_vca_path()
    if not os.path.exists(vca_path):
        raise FileNotFoundError(f"VCA result file not found: {vca_path}")

    import scipy.io as sio

    mat = sio.loadmat(vca_path)
    if "endmembers" not in mat:
        raise KeyError(f"'endmembers' not found in VCA result file: {vca_path}")
    return _format_endmembers(mat["endmembers"], num_bands=num_bands)

class Bottleneck(nn.Module):
    def __init__(self, channels, bottleneck_ratio=4, kernel_size=3):
        super().__init__()
        mid_channels = max(channels // bottleneck_ratio, 1)
        padding = kernel_size // 2

        self.body = nn.Sequential(
            # 1x1 降维
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 3x3 卷积
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            # 1x1 升维
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.body(x)
        out = out + x          
        return self.act(out)

class LightUnmixContextEncoder(nn.Module):
    def __init__(self, in_channels, context_dim=128, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(min(in_channels, 128), 32)

        self.in_channels = int(in_channels)
        self.out_channels = int(hidden_channels)

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
        )
        self.act = nn.GELU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )

    def forward(self, x):
        base = self.input_proj(x)
        feature = self.act(base + self.refine(base))
        avg_context = self.avg_pool(feature).flatten(1)
        max_context = self.max_pool(feature).flatten(1)
        context = self.proj(torch.cat([avg_context, max_context], dim=1))
        return context, feature


class EndmemberTokenizer(nn.Module):
    def __init__(self, num_bands, token_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_bands, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
            nn.LayerNorm(token_dim),
        )

    def forward(self, endmembers):
        return self.net(endmembers)


class EndmemberBundleAttention(nn.Module):
    def __init__(self, token_dim=64, num_layers=1, num_heads=4, dropout=0.0):
        super().__init__()
        if token_dim % num_heads != 0:
            valid_heads = [
                h for h in range(min(int(num_heads), token_dim), 0, -1)
                if token_dim % h == 0
            ]
            num_heads = valid_heads[0]

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                dim_feedforward=token_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(max(int(num_layers), 0))
        ])
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, tokens):
        for layer in self.layers:
            tokens = layer(tokens)
        return self.norm(tokens)


class VariationalRegionalEndmemberAdapter(nn.Module):
    def __init__(
        self,
        num_bands,
        num_endmembers,
        context_dim=128,
        token_dim=64,
        latent_dim=16,
        alpha=0.1,
        bundle_layers=1,
        bundle_heads=4,
        bundle_dropout=0.0,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.latent_dim = int(latent_dim)
        self.alpha = float(alpha)

        self.tokenizer = EndmemberTokenizer(num_bands, token_dim=token_dim)
        self.bundle_attention = EndmemberBundleAttention(
            token_dim=token_dim,
            num_layers=bundle_layers,
            num_heads=bundle_heads,
            dropout=bundle_dropout,
        )
        cond_dim = token_dim + context_dim

        self.context_fusion = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        self.variational_encoder = nn.Sequential(
            nn.Linear(cond_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, self.latent_dim * 2),
        )
        self.delta_decoder = nn.Sequential(
            nn.Linear(token_dim + self.latent_dim, token_dim * 2),
            nn.GELU(),
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_bands),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(cond_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, num_bands),
            nn.Sigmoid(),
        )

    @staticmethod
    def _reparameterize(mu, logvar, sample=True):
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, global_endmembers, context, spectral_context, sample=True):
        if context.ndim != 2:
            raise ValueError(f"context must be [B,D], got {tuple(context.shape)}.")
        if spectral_context.shape != context.shape:
            raise ValueError(
                "spectral_context must match context shape, "
                f"got {tuple(spectral_context.shape)} vs {tuple(context.shape)}."
            )
        if global_endmembers.ndim == 2:
            global_endmembers = global_endmembers.unsqueeze(0).expand(context.shape[0], -1, -1)
        if global_endmembers.ndim != 3:
            raise ValueError(
                f"Expected global_endmembers as [K, B] or [B, K, B], got {tuple(global_endmembers.shape)}."
            )
        if global_endmembers.shape[1] != self.num_endmembers or global_endmembers.shape[2] != self.num_bands:
            raise ValueError(
                f"Expected endmembers [batch, {self.num_endmembers}, {self.num_bands}], "
                f"got {tuple(global_endmembers.shape)}."
            )
        if global_endmembers.shape[0] != context.shape[0]:
            raise ValueError("global_endmembers and context must have the same batch size.")

        batch_size = context.shape[0]
        tokens = self.tokenizer(global_endmembers)
        tokens = self.bundle_attention(tokens)
        regional_context = self.context_fusion(torch.cat([context, spectral_context], dim=-1))
        context_tokens = regional_context.unsqueeze(1).expand(batch_size, self.num_endmembers, -1)
        cond = torch.cat([tokens, context_tokens], dim=-1)

        latent_stats = self.variational_encoder(cond)
        latent_mu, latent_logvar = latent_stats.chunk(2, dim=-1)
        latent_logvar = latent_logvar.clamp(-10.0, 10.0)
        latent = self._reparameterize(latent_mu, latent_logvar, sample=sample)

        delta = self.delta_decoder(torch.cat([tokens, latent], dim=-1))
        residual_gate = self.gate_head(cond)
        regional = global_endmembers + self.alpha * residual_gate * torch.tanh(delta)
        regional = regional.clamp(0.0, 1.0)
        kl_loss = standard_normal_kl_loss(latent_mu, latent_logvar)
        return (
            regional,
            delta,
            residual_gate,
            regional_context,
            latent_mu,
            latent_logvar,
            latent,
            kl_loss,
        )


class SimilarityAbundanceEstimator(nn.Module):
    def __init__(self, num_bands, num_endmembers, feature_channels=None, hidden_channels=64, temperature=0.1):
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.temperature = float(temperature)

        self.hsi_proj = nn.Sequential(
            nn.Conv2d(num_bands, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        )

        self.feature_proj = None
        extra_channels = hidden_channels
        if feature_channels is not None:
            self.feature_proj = nn.Sequential(
                nn.Conv2d(feature_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.GELU(),
            )
            extra_channels += hidden_channels

        head_in = num_endmembers + extra_channels
        self.head = nn.Sequential(
            nn.Conv2d(head_in, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, num_endmembers, kernel_size=1),
        )

    def spectral_similarity(self, hsi_patch, endmembers, eps=1e-8):
        b, c, h, w = hsi_patch.shape
        pixels = hsi_patch.permute(0, 2, 3, 1).reshape(b, h * w, c)
        pixels = F.normalize(pixels, dim=-1, eps=eps)
        endmembers = F.normalize(endmembers, dim=-1, eps=eps)
        similarity = torch.einsum("bnc,bkc->bnk", pixels, endmembers)
        return similarity.permute(0, 2, 1).reshape(b, self.num_endmembers, h, w)

    def forward(self, hsi_patch, regional_endmembers, unmix_feature=None):
        similarity = self.spectral_similarity(hsi_patch, regional_endmembers)
        features = [similarity, self.hsi_proj(hsi_patch)]
        if self.feature_proj is not None and unmix_feature is not None:
            features.append(self.feature_proj(unmix_feature))
        logits = self.head(torch.cat(features, dim=1))
        logits = logits + similarity / max(self.temperature, 1e-6)
        abundance = torch.softmax(logits, dim=1)
        reconstruction = torch.einsum("bkc,bkhw->bchw", regional_endmembers, abundance)
        return abundance, reconstruction, similarity


class HSIUnmixingBranch(nn.Module):
    """
    HSI-only unmixing branch:
      1. load or receive global VCA endmembers E_g
      2. encode shared shallow HSI features as regional context
      3. use a single-encoder VAE to adapt VCA anchors to regional endmembers E_r
      4. estimate abundance A from raw-spectral similarity between X and E_r
      5. reconstruct the raw spectrum X_hat = A E_r

    LiDAR constraints/refinement are intentionally not included here.
    """
    def __init__(
        self,
        hsi_channels,
        num_endmembers=None,
        vca_path=None,
        context_dim=128,
        token_dim=64,
        latent_dim=16,
        alpha=0.1,
        bundle_layers=1,
        bundle_heads=4,
        bundle_dropout=0.0,
        context_input_channels=None,
        context_hidden_channels=None,
        abundance_hidden_channels=64,
        abundance_temperature=0.1,
        anchor_mse_weight=1.0,
        anchor_sam_weight=1.0,
        abundance_sparse_eps=1e-8,
        load_vca=True,
    ):
        super().__init__()
        loaded_endmembers = None
        if load_vca:
            try:
                loaded_endmembers = load_vca_endmembers(vca_path=vca_path, num_bands=hsi_channels)
            except FileNotFoundError:
                loaded_endmembers = None

        if num_endmembers is None:
            if loaded_endmembers is not None:
                num_endmembers = loaded_endmembers.shape[0]
            else:
                num_endmembers = int(globals().get("vca_num_endmembers", 10))

        self.hsi_channels = hsi_channels
        self.num_endmembers = num_endmembers
        self.context_dim = context_dim
        self.context_input_channels = int(context_input_channels or hsi_channels)
        self.anchor_mse_weight = float(anchor_mse_weight)
        self.anchor_sam_weight = float(anchor_sam_weight)
        self.abundance_sparse_eps = float(abundance_sparse_eps)

        self.context_encoder = LightUnmixContextEncoder(
            self.context_input_channels,
            context_dim=context_dim,
            hidden_channels=context_hidden_channels,
        )
        self.spectral_summary = nn.Sequential(
            nn.Linear(hsi_channels, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )
        self.endmember_adapter = VariationalRegionalEndmemberAdapter(
            num_bands=hsi_channels,
            num_endmembers=num_endmembers,
            context_dim=context_dim,
            token_dim=token_dim,
            latent_dim=latent_dim,
            alpha=alpha,
            bundle_layers=bundle_layers,
            bundle_heads=bundle_heads,
            bundle_dropout=bundle_dropout,
        )
        unmix_feature_channels = self.context_encoder.out_channels
        self.abundance_estimator = SimilarityAbundanceEstimator(
            num_bands=hsi_channels,
            num_endmembers=num_endmembers,
            feature_channels=unmix_feature_channels,
            hidden_channels=abundance_hidden_channels,
            temperature=abundance_temperature,
        )

        if loaded_endmembers is not None:
            if loaded_endmembers.shape[0] != num_endmembers:
                raise ValueError(
                    f"Loaded {loaded_endmembers.shape[0]} endmembers, but num_endmembers={num_endmembers}."
                )
            self.register_buffer("global_endmembers", loaded_endmembers, persistent=True)
        else:
            self.register_buffer("global_endmembers", torch.empty(0), persistent=True)

    def set_global_endmembers(self, endmembers):
        endmembers = _format_endmembers(endmembers, num_bands=self.hsi_channels)
        if endmembers.shape[0] != self.num_endmembers:
            raise ValueError(
                f"Expected {self.num_endmembers} endmembers, got {endmembers.shape[0]}."
            )
        self.global_endmembers = endmembers.to(self.global_endmembers.device)

    def _get_global_endmembers(self, endmembers, device):
        if endmembers is None:
            if self.global_endmembers.numel() == 0:
                raise RuntimeError(
                    "No global VCA endmembers are loaded. Pass endmembers to forward() "
                    "or call set_global_endmembers()."
                )
            endmembers = self.global_endmembers
        endmembers = _format_endmembers(endmembers, num_bands=self.hsi_channels).to(device)
        if endmembers.shape[0] != self.num_endmembers:
            raise ValueError(
                f"Expected {self.num_endmembers} endmembers, got {endmembers.shape[0]}."
            )
        return endmembers

    def forward(
        self,
        hsi_patch,
        hsi_feature=None,
        global_endmembers=None,
        sample=None,
    ):
        if hsi_patch.ndim != 4:
            raise ValueError(f"Expected hsi_patch [B, C, H, W], got {tuple(hsi_patch.shape)}.")
        if hsi_patch.shape[1] != self.hsi_channels:
            raise ValueError(
                f"Expected {self.hsi_channels} HSI channels, got {hsi_patch.shape[1]}."
            )

        if hsi_feature is None:
            if self.context_input_channels != self.hsi_channels:
                raise ValueError(
                    "hsi_feature is required when the unmixing context encoder "
                    "uses shared feature channels."
                )
            context_input = hsi_patch
        else:
            if hsi_feature.ndim != 4:
                raise ValueError(
                    f"Expected hsi_feature [B, C_f, H, W], got {tuple(hsi_feature.shape)}."
                )
            if hsi_feature.shape[1] != self.context_input_channels:
                raise ValueError(
                    f"Expected {self.context_input_channels} shared HSI feature channels, "
                    f"got {hsi_feature.shape[1]}."
                )
            if (
                hsi_feature.shape[0] != hsi_patch.shape[0]
                or hsi_feature.shape[2:] != hsi_patch.shape[2:]
            ):
                raise ValueError("hsi_feature and hsi_patch must share batch size and spatial resolution.")
            context_input = hsi_feature
        context, unmix_feature = self.context_encoder(context_input)

        mean_spectrum = hsi_patch.mean(dim=(2, 3))
        spectral_context = self.spectral_summary(mean_spectrum)
        if sample is None:
            sample = self.training

        endmembers = self._get_global_endmembers(global_endmembers, hsi_patch.device)
        (
            regional_endmembers,
            delta_endmembers,
            residual_gate,
            regional_context,
            latent_mu,
            latent_logvar,
            latent,
            kl_loss,
        ) = self.endmember_adapter(
            endmembers,
            context=context,
            spectral_context=spectral_context,
            sample=sample,
        )

        abundance, reconstruction, similarity = self.abundance_estimator(
            hsi_patch,
            regional_endmembers,
            unmix_feature=unmix_feature,
        )
        anchor_loss = endmember_anchor_loss(
            regional_endmembers,
            endmembers,
            mse_weight=self.anchor_mse_weight,
            sam_weight=self.anchor_sam_weight,
        )
        abundance_sparse_loss = abundance_sparsity_loss(abundance, eps=self.abundance_sparse_eps)

        return UnmixingOutput(
            regional_endmembers=regional_endmembers,
            abundance=abundance,
            reconstruction=reconstruction,
            similarity=similarity,
            delta_endmembers=delta_endmembers,
            residual_gate=residual_gate,
            regional_context=regional_context,
            latent_mu=latent_mu,
            latent_logvar=latent_logvar,
            latent=latent,
            kl_loss=kl_loss,
            anchor_loss=anchor_loss,
            abundance_sparse_loss=abundance_sparse_loss,
        )
