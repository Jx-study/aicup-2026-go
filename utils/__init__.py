"""工具函式套件。"""
import importlib.util
from pathlib import Path

# 轉發根目錄官方工具模組定義的所有屬性
_root_utils_path = Path(__file__).resolve().parent.parent / 'utils.py'
if _root_utils_path.exists():
    _spec = importlib.util.spec_from_file_location('_root_utils', str(_root_utils_path))
    if _spec and _spec.loader:
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        for _attr in dir(_mod):
            if not _attr.startswith('__'):
                globals()[_attr] = getattr(_mod, _attr)
