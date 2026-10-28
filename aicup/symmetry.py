"""盤面的 8 種對稱變換（4 種旋轉 × 是否左右鏡射），作用在最後兩維。

k 為 0–7：先逆時針旋轉 (k % 4) 次 90 度，k >= 4 時再左右鏡射。k = 0 是恆等變換。
同時支援 torch tensor 與 numpy 陣列，形狀如 (B, C, 19, 19) 或 (19, 19)。
"""
import numpy as np

N_SYMMETRIES = 8


def _check(k):
    if not 0 <= k < N_SYMMETRIES:
        raise ValueError('k 必須在 0 到 %d 之間，收到 %r' % (N_SYMMETRIES - 1, k))


def _ops(x):
    if hasattr(x, 'rot90'):  # torch tensor
        import torch
        return (lambda a, r: torch.rot90(a, r, dims=(-2, -1)),
                lambda a: torch.flip(a, dims=(-1,)), lambda a: a)
    return (lambda a, r: np.rot90(a, r, axes=(-2, -1)),
            lambda a: np.flip(a, axis=-1),
            np.ascontiguousarray)  # 讓輸出可直接交給 torch.from_numpy


def apply(x, k):
    """套用第 k 種對稱變換。"""
    _check(k)
    rot, flip, finish = _ops(x)
    y = rot(x, k % 4)
    return finish(flip(y) if k >= 4 else y)


def invert(x, k):
    """套用第 k 種變換的反變換，invert(apply(x, k), k) == x。"""
    _check(k)
    rot, flip, finish = _ops(x)
    y = flip(x) if k >= 4 else x
    return finish(rot(y, -(k % 4)))
