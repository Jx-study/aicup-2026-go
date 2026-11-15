"""KataGo 1-visit 評估的逐局失誤特徵。

score_loss、winrate_loss 是 analyze_humansl --eval-visits 輸出的每手目數損失與勝率損失，
正值表示比 KataGo 最佳手差；1-visit 評估有雜訊，所以也有負值。
"""
import numpy as np

BLUNDER_POINTS = (2.0, 5.0, 15.0)  # 目數損失門檻
WINRATE_BLUNDER = 0.1  # 勝率損失門檻
LOSS_COLUMNS = (('mean_loss', 'mean_pos_loss', 'q90_loss')
                + tuple('blunder_rate_%g' % t for t in BLUNDER_POINTS)
                + ('winrate_drop_mean', 'winrate_blunder_rate'))
N_LOSS = len(LOSS_COLUMNS)


def _finite(values):
    v = np.array([np.nan if x is None else x for x in values], dtype=np.float64)
    return v[np.isfinite(v)]


def game_loss_features(row):
    """一局一側的失誤特徵，回傳 (N_LOSS,)，欄位依 LOSS_COLUMNS。

    None 與 NaN 的手先去掉。沒有 score_loss 欄位或沒有可用的手時，目數相關欄位為 NaN；
    winrate_loss 同理，只影響最後兩欄。
    """
    out = np.full(N_LOSS, np.nan)
    sl = _finite(row.get('score_loss') or [])
    if len(sl):
        out[0] = sl.mean()
        out[1] = np.maximum(sl, 0).mean()
        out[2] = np.quantile(sl, 0.9)
        for i, t in enumerate(BLUNDER_POINTS):
            out[3 + i] = float((sl > t).mean())
    wl = _finite(row.get('winrate_loss') or [])
    if len(wl):
        out[-2] = np.maximum(wl, 0).mean()
        out[-1] = float((wl > WINRATE_BLUNDER).mean())
    return out
