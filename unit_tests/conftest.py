import os

# 本機 anaconda 的 numpy 與 torch 各帶一份 OpenMP，不設定會在 import torch 時中止
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
