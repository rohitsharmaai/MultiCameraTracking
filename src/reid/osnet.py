"""OSNet x1.0 body ReID.

Exact artifact:
osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth

Source: https://huggingface.co/kaiyangzhou/osnet (MIT weights).
Architecture and module names match torchreid so that checkpoint loads.
Do not silently substitute another OSNet size or a third-party re-upload.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import ModelRegistry
from src.reid.normalize import l2_normalize

logger = logging.getLogger(__name__)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding,
            bias=False, groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, stride=stride, padding=0, bias=False, groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False, groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_channels, num_gates=None, return_gates=False, gate_activation="sigmoid", reduction=16):
        super().__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates, kernel_size=1, bias=True, padding=0)
        self.gate_activation = nn.Sigmoid() if gate_activation == "sigmoid" else None

    def forward(self, x):
        input_ = x
        x = self.global_avgpool(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input_ * x


class OSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_reduction=4):
        super().__init__()
        mid = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1(in_channels, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(
            LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid),
        )
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(x3 + identity, inplace=True)


class OSNet(nn.Module):
    """OSNet x1.0 with torchreid module names (conv2/3 include spatial reduction)."""

    def __init__(self, feature_dim: int = 512):
        super().__init__()
        channels = [64, 256, 384, 512]
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(2, channels[0], channels[1], reduce_spatial_size=True)
        self.conv3 = self._make_layer(2, channels[1], channels[2], reduce_spatial_size=True)
        self.conv4 = self._make_layer(2, channels[2], channels[3], reduce_spatial_size=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(feature_dim, channels[3])
        self.classifier = nn.Linear(feature_dim, 1)

    @staticmethod
    def _make_layer(num: int, in_c: int, out_c: int, reduce_spatial_size: bool) -> nn.Sequential:
        layers: list[nn.Module] = [OSBlock(in_c, out_c)]
        for _ in range(1, num):
            layers.append(OSBlock(out_c, out_c))
        if reduce_spatial_size:
            layers.append(nn.Sequential(Conv1x1(out_c, out_c), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    @staticmethod
    def _construct_fc_layer(feature_dim: int, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _load_torch(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class OSNetEncoder:
    name = "OSNet-x1.0"
    version = "msmt17-combineall-ep150"
    input_size = (128, 256)  # (W, H)

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        device: Optional[str] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model
        self._path = None
        if registry is not None:
            entry = registry.require_loadable("OSNet-x1.0")
            self.version = entry.version
            self._path = registry.resolve_path(entry)

    def load(self) -> None:
        if self._model is not None:
            return
        if self._path is None or not self._path.exists():
            raise FileNotFoundError(
                "OSNet x1.0 checkpoint missing. Download the exact artifact from "
                "https://huggingface.co/kaiyangzhou/osnet"
            )
        model = OSNet(feature_dim=512)
        state = _load_torch(str(self._path))
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError("OSNet checkpoint is not a state_dict")
        cleaned = {}
        for k, v in state.items():
            key = k[7:] if k.startswith("module.") else k
            if key.startswith("classifier"):
                continue
            cleaned[key] = v
        if not any(k.startswith("fc.") for k in cleaned):
            model.fc = nn.Identity()
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        # Classifier is training-only; we drop those keys on purpose.
        missing = [k for k in missing if not k.startswith("classifier")]
        if missing:
            logger.error("OSNet missing keys after load: %s", list(missing)[:12])
        if unexpected:
            logger.warning("OSNet unexpected keys (ignored): %s", list(unexpected)[:12])
        model.eval()
        self._model = model.to(self.device)

    def embed(self, crop: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self._model is None:
            self.load()
        assert self._model is not None
        image = crop
        if mask is not None and mask.shape[:2] == image.shape[:2]:
            image = image.copy()
            image[mask == 0] = 0
        inp = cv2.resize(image, self.input_size)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        inp = (inp - mean) / std
        tensor = torch.from_numpy(np.transpose(inp, (2, 0, 1))[None, ...]).to(self.device)
        with torch.no_grad():
            feat = self._model(tensor)
            if feat.ndim > 1:
                feat = feat.reshape(feat.shape[0], -1)[0]
            feat = feat.cpu().numpy()
        return l2_normalize(feat)
