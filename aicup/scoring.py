"""
棋力預測的官方評分與期望得分決策。
只依賴 numpy，不需要 GPU 或資料集，可在本機與 Colab 共用。
"""
import numpy as np

# 由弱到強，與 rank-prediction.ipynb 的 LEVELS 一致；評分的「相鄰一級」以此順序判定
LEVELS = ['D', 'C', 'B', 'A', '1D', '2D', '3D', '4D', '5D', '6D']
RANK_TO_ID = {lv: i for i, lv in enumerate(LEVELS)}

NEAR_SCORE = float(np.exp(-1))  # 差一級的得分 1/e


def _to_ids(levels):
    """等級字串或整數 index 一律轉成 index 陣列。"""
    arr = np.asarray(levels)
    if arr.dtype.kind in 'iu':
        return arr.astype(np.int64)
    return np.array([RANK_TO_ID[str(lv)] for lv in arr], dtype=np.int64)


def score_strength(y_true, y_pred):
    """官方評分：猜中 1 分，差一級 1/e，其餘 0，取平均。

    y_true、y_pred 可以是等級字串（'3D'）或 index（0–9）。
    """
    t, p = _to_ids(y_true), _to_ids(y_pred)
    if t.shape != p.shape:
        raise ValueError('y_true 與 y_pred 長度不同：%s vs %s' % (t.shape, p.shape))
    diff = np.abs(t - p)
    scores = np.where(diff == 0, 1.0, np.where(diff == 1, NEAR_SCORE, 0.0))
    return float(scores.mean())


def expected_scores(probs):
    """每個候選等級的期望得分：p[k] + (1/e)·(p[k-1] + p[k+1])。

    probs 形狀為 (N, 10) 或 (10,)，每列是一題的等級機率分布。
    兩端等級只有一個鄰居，缺的一側視為 0。
    """
    p = np.asarray(probs, dtype=np.float64)
    neighbors = np.zeros_like(p)
    neighbors[..., 1:] += p[..., :-1]   # 下一級
    neighbors[..., :-1] += p[..., 1:]   # 上一級
    return p + NEAR_SCORE * neighbors


def pick_level(probs):
    """回傳期望得分最大的等級 index，形狀為 (N,) 或純量。"""
    return np.argmax(expected_scores(probs), axis=-1)


def aggregate_games(game_probs, method='mean'):
    """把同一題多局的機率合成一個分布。

    game_probs 形狀為 (G, 10)。
    'mean'：機率平均，與教學 notebook 相同。
    'log'：各局機率相乘後重新正規化（log 相加）。訓練集等級均衡，
    先驗是均勻分布，在各局獨立的假設下這就是後驗機率。
    """
    p = np.asarray(game_probs, dtype=np.float64)
    if method == 'mean':
        return p.mean(axis=0)
    if method == 'log':
        logp = np.log(np.clip(p, 1e-12, None)).sum(axis=0)
        logp -= logp.max()  # 避免 exp 下溢
        q = np.exp(logp)
        return q / q.sum()
    raise ValueError('未知的 method：%s' % method)


def apply_temperature(probs, temp):
    """每局機率改為 softmax(log p / temp)。

    temp > 1 縮小各等級機率的差距，temp < 1 放大差距，最高的等級不變。
    """
    if temp <= 0:
        raise ValueError('溫度要大於 0，收到 %g' % temp)
    logp = np.log(np.clip(np.asarray(probs, np.float64), 1e-12, None)) / temp
    logp -= logp.max(axis=-1, keepdims=True)
    q = np.exp(logp)
    return q / q.sum(axis=-1, keepdims=True)
