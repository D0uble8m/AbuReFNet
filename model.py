from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from HU import HSIUnmixingBranch
from model_component import (
    DSMStem,
    HSIStem,
    LiDARSecondaryEncoder,
    AbundanceGuidedHSIEnhancement,
    UncertaintyAwareBoundaryFusion,
)


ModelOutput = namedtuple(
    "ModelOutput",
    [
        "logits",
        "fused_feature",
        "hsi_feature",
        "lidar_feature",
        "unmixing_output",
        "hsi_enhancement_output",
        "lidar_output",
        "fusion_output",
    ],
)


class GlobalPoolClassifier(nn.Module):
    """Classify a patch from its globally averaged feature representation."""
    def __init__(self, channels, num_classes, hidden_channels=None, dropout=0.1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(channels, 32)
        self.classifier = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, num_classes),
        )

    def forward(self, feature):
        if feature.ndim != 4:
            raise ValueError(f"feature must be [B,C,H,W], got {tuple(feature.shape)}.")
        global_feature = F.adaptive_avg_pool2d(feature, output_size=1).flatten(1)
        return self.classifier(global_feature)


class UnmixingGuidedHSILiDARClassifier(nn.Module):
    def __init__(
        self,
        hsi_channels,
        num_classes,
        num_endmembers=None,
        vca_path=None,
        feature_channels=64,
        hsi_kernel_size=3,
        dsm_mid_channels=None,
        dsm_kernel_size=3,
        mamba_kwargs=None,
        gcn_kwargs=None,
        hsi_enhancement_dropout=0.0,
        lidar_encoder_kwargs=None,
        fusion_channels=None,
        fusion_dropout=0.0,
        classifier_hidden_channels=None,
        classifier_dropout=0.1,
        detach_abundance=False,
        unmixing_kwargs=None,
    ):
        super().__init__()
        self.hsi_channels = int(hsi_channels)
        self.num_classes = int(num_classes)
        self.feature_channels = int(feature_channels)
        self.detach_abundance = bool(detach_abundance)

        unmixing_kwargs = dict(unmixing_kwargs or {})
        unmixing_kwargs.setdefault("context_input_channels", self.feature_channels)
        self.unmixing_branch = HSIUnmixingBranch(
            hsi_channels=self.hsi_channels,
            num_endmembers=num_endmembers,
            vca_path=vca_path,
            **unmixing_kwargs,
        )
        self.num_endmembers = self.unmixing_branch.num_endmembers

        # This shallow HSI representation is shared by unmixing context
        # encoding and the downstream classification backbone.
        self.hsi_input_proj = nn.Sequential(
            nn.Conv2d(self.hsi_channels, self.feature_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.feature_channels),
            nn.GELU(),
        )
        self.hsi_stem = HSIStem(
            self.feature_channels,
            kernel_size=hsi_kernel_size,
        )
        self.hsi_enhancement = AbundanceGuidedHSIEnhancement(
            channels=self.feature_channels,
            num_endmembers=self.num_endmembers,
            mamba_kwargs=mamba_kwargs,
            gcn_kwargs=gcn_kwargs,
            dropout=hsi_enhancement_dropout,
        )

        self.dsm_stem = DSMStem(
            hsi_channels=self.feature_channels,
            mid_channels=dsm_mid_channels,
            kernel_size=dsm_kernel_size,
        )
        self.lidar_encoder = LiDARSecondaryEncoder(
            channels=self.feature_channels,
            **dict(lidar_encoder_kwargs or {}),
        )

        fusion_channels = int(fusion_channels or self.feature_channels)
        self.fusion = UncertaintyAwareBoundaryFusion(
            hsi_channels=self.feature_channels,
            lidar_channels=self.feature_channels,
            num_endmembers=self.num_endmembers,
            fusion_channels=fusion_channels,
            dropout=fusion_dropout,
        )
        self.classifier = GlobalPoolClassifier(
            channels=fusion_channels,
            num_classes=self.num_classes,
            hidden_channels=classifier_hidden_channels,
            dropout=classifier_dropout,
        )

    @staticmethod
    def _prepare_lidar(lidar):
        if lidar.ndim == 3:
            lidar = lidar.unsqueeze(1)
        if lidar.ndim != 4:
            raise ValueError(f"lidar must be [B,H,W] or [B,1,H,W], got {tuple(lidar.shape)}.")
        if lidar.shape[1] != 1:
            raise ValueError(f"Expected one LiDAR/DSM channel, got {lidar.shape[1]}.")
        return lidar

    def forward(
        self,
        hsi,
        lidar,
        global_endmembers=None,
        sample=None,
        detach_abundance=None,
    ):
        if hsi.ndim != 4:
            raise ValueError(f"hsi must be [B,C,H,W], got {tuple(hsi.shape)}.")
        if hsi.shape[1] != self.hsi_channels:
            raise ValueError(f"Expected {self.hsi_channels} HSI bands, got {hsi.shape[1]}.")
        lidar = self._prepare_lidar(lidar)
        if lidar.shape[0] != hsi.shape[0] or lidar.shape[2:] != hsi.shape[2:]:
            raise ValueError("HSI and LiDAR inputs must share batch size and spatial resolution.")

        if detach_abundance is None:
            detach_abundance = self.detach_abundance

        shallow_hsi = self.hsi_stem(self.hsi_input_proj(hsi))
        unmixing_output = self.unmixing_branch(
            hsi,
            hsi_feature=shallow_hsi,
            global_endmembers=global_endmembers,
            sample=sample,
        )

        hsi_enhancement_output = self.hsi_enhancement(
            shallow_hsi,
            unmixing_output.abundance,
            detach_abundance=detach_abundance,
        )
        hsi_feature = hsi_enhancement_output.feature

        shallow_lidar = self.dsm_stem(lidar)
        lidar_output = self.lidar_encoder(shallow_lidar)
        lidar_feature = lidar_output.feature

        fusion_output = self.fusion(
            hsi_feature,
            lidar_feature,
            unmixing_output.abundance,
            detach_abundance=detach_abundance,
            lidar_raw=lidar,
        )
        fused_feature = fusion_output.feature
        logits = self.classifier(fused_feature)

        return ModelOutput(
            logits=logits,
            fused_feature=fused_feature,
            hsi_feature=hsi_feature,
            lidar_feature=lidar_feature,
            unmixing_output=unmixing_output,
            hsi_enhancement_output=hsi_enhancement_output,
            lidar_output=lidar_output,
            fusion_output=fusion_output,
        )

