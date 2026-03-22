import torch
import torch.nn as nn

class ScaledSigmoid(nn.Module):
    def __init__(self, scale=1.1, shift=0.05):
        """
        Formula: out = scale * sigmoid(x) + shift
        Note: To achieve the YOLOv4 effect of "eliminating hard-to-reach boundaries",
        the shift is typically negative.
        e.g., scale=1.1, shift=-0.05 or scale=2.0, shift=-0.5
        Parameters are opened up here for various (scale, shift) experiments.
        """
        super(ScaledSigmoid, self).__init__()
        self.scale = scale
        self.shift = shift
    def forward(self, x):
        return self.scale * torch.sigmoid(x) + self.shift
