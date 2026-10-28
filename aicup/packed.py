"""
棋力特徵的 bit 打包格式：本機 CPU 打包存檔，訓練時整份載入 GPU 再解包。
特徵是 0/1 遮罩，每格 1 bit。一局 (500, 19, 19) = 180,500 bit，打包後 22,563 byte。

分片檔 {tag}-w{i}-{n:06d}.npz（np.savez_compressed）內含：
    features   (N, PACKED_LEN) uint8，np.packbits 預設的 big bit order
    rank_idx   (N,) int8，LEVELS 的 index
    color      (N,) uint8，目標玩家執色，0 黑 1 白
    key        (N,) str，'{rank}_{game_id}'
    player_id  (N,) str
"""
import glob
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

FEATURE_SHAPE = (500, 19, 19)  # 500 channels = 最後 250 個盤面 × 黑白兩面
N_BITS = int(np.prod(FEATURE_SHAPE))
PACKED_LEN = (N_BITS + 7) // 8
FIELDS = ('features', 'rank_idx', 'color', 'key', 'player_id')


def pack(feat):
    """(500, 19, 19) 的 0/1 陣列打包成 (PACKED_LEN,) uint8。"""
    if feat.shape != FEATURE_SHAPE:
        raise ValueError('特徵形狀 %s，預期 %s' % (feat.shape, FEATURE_SHAPE))
    return np.packbits(feat.reshape(-1).astype(bool))


def unpack_numpy(packed):
    """(N, PACKED_LEN) uint8 還原成 (N, 500, 19, 19) uint8，供測試與 CPU 推論使用。"""
    bits = np.unpackbits(packed, axis=-1, count=N_BITS)
    return bits.reshape(-1, *FEATURE_SHAPE)


def unpack_torch(packed, dtype=None):
    """GPU 上解包：(B, PACKED_LEN) uint8 tensor 還原成 (B, 500, 19, 19)。"""
    import torch
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)  # big bit order
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    bits = bits.reshape(packed.shape[0], -1)[:, :N_BITS]
    return bits.reshape(-1, *FEATURE_SHAPE).to(dtype or torch.float32)


def save_shard(path, rows):
    """rows 是 dict 的 list，鍵為 FIELDS；先寫暫存檔再改名，中斷不留半成品。"""
    arrays = {
        'features': np.stack([r['features'] for r in rows]) if rows
        else np.zeros((0, PACKED_LEN), np.uint8),
        'rank_idx': np.array([r['rank_idx'] for r in rows], np.int8),
        'color': np.array([r['color'] for r in rows], np.uint8),
        'key': np.array([r['key'] for r in rows], dtype=str),
        'player_id': np.array([r['player_id'] for r in rows], dtype=str),
    }
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:  # 傳檔案物件，避免 numpy 自動補 .npz 副檔名
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def list_shards(split_dir):
    return sorted(glob.glob(os.path.join(split_dir, '*.npz')))


def load_split(split_dir, workers=None):
    """把一個 split 的所有分片載入，回傳 FIELDS 對應的 numpy 陣列。

    features 先依總筆數配置再逐片填入，記憶體只保留一份。
    解壓由 zlib 執行、不受 GIL 限制，用 thread 平行讀檔。
    """
    paths = list_shards(split_dir)
    if not paths:
        raise FileNotFoundError('%s 下沒有 .npz 分片' % split_dir)

    def read_meta(p):
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in FIELDS if k != 'features'}

    with ThreadPoolExecutor(workers or os.cpu_count()) as ex:
        metas = list(ex.map(read_meta, paths))
    sizes = [len(m['rank_idx']) for m in metas]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    features = np.empty((offsets[-1], PACKED_LEN), np.uint8)

    def fill(i):
        with np.load(paths[i], allow_pickle=False) as z:
            features[offsets[i]:offsets[i + 1]] = z['features']

    with ThreadPoolExecutor(workers or os.cpu_count()) as ex:
        list(ex.map(fill, range(len(paths))))
    out = {k: np.concatenate([m[k] for m in metas]) for k in FIELDS if k != 'features'}
    out['features'] = features
    return out
