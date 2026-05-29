import torch
import torch.nn as nn
import torchvision.transforms as TF
import torchvision.transforms.functional as TF_F


class ToTensor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        assert x.dtype == torch.uint8
        x = x.to(torch.float32) / 255.0
        return x


class Pad(nn.Module):
    def __init__(self, padding, fill=0, padding_mode='constant'):
        super().__init__()
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode
        self.pad = TF.Pad(padding=tuple(padding), fill=fill, padding_mode=padding_mode)
    
    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Can only pad tensor of 4 dims."
        return self.pad(x)


class TemporalConsistentColorJitter(nn.Module):
    """Apply the same color jitter to all frames in a video clip.

    Expects input shape [T, C, H, W] (float32, range [0, 1]).
    Applies identical random brightness, contrast, saturation, and hue
    perturbations across all T frames to maintain temporal consistency.
    """

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.02,
        p: float = 0.5,
    ):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        assert x.ndim == 4, f"Expected [T, C, H, W], got shape {x.shape}"

        # Sample jitter params once, apply to all frames
        brightness_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
        contrast_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
        saturation_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.saturation
        hue_factor = (torch.rand(1).item() * 2 - 1) * self.hue

        x = TF_F.adjust_brightness(x, brightness_factor)
        x = TF_F.adjust_contrast(x, contrast_factor)
        x = TF_F.adjust_saturation(x, saturation_factor)
        x = TF_F.adjust_hue(x, hue_factor)
        return x.clamp(0.0, 1.0)


class TemporalConsistentGaussianNoise(nn.Module):
    """Add small Gaussian noise to video frames during training.

    Helps prevent the model from over-relying on exact pixel values.
    Noise is sampled independently per frame (simulates sensor noise).
    Expects input shape [T, C, H, W] (float32, range [0, 1]).
    """

    def __init__(self, std: float = 0.02, p: float = 0.3):
        super().__init__()
        self.std = std
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        noise = torch.randn_like(x) * self.std
        return (x + noise).clamp(0.0, 1.0)
