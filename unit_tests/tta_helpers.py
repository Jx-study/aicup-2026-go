import torch

from aicup import symmetry
from aicup.scoring import LEVELS
from network import GoRankResNet


def sharp_model(in_channels=500, scale=50.0):
    """輸出很尖的小模型，讓各對稱方向的機率差異夠大，平均 logits 與平均機率才分得開。"""
    torch.manual_seed(0)
    model = GoRankResNet(in_channels=in_channels, num_channels=8, num_blocks=1, num_classes=len(LEVELS))
    with torch.no_grad():
        model.fc_2.weight.mul_(scale)
    return model.eval()


def tta_expect(model, x):
    """參照實作：8 種對稱方向各自 softmax 後取平均。"""
    with torch.no_grad():
        return torch.stack([torch.softmax(model(symmetry.apply(x, k).contiguous()), 1)
                            for k in range(symmetry.N_SYMMETRIES)]).mean(0)
