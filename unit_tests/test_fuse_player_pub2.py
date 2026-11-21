"""fuse_player pub2/priv2 專用快取與參數覆寫之單元測試。"""

import json

import pytest

from aicup.fuse_player import _load_rank_cache, run_full_fused_pipeline


def test_load_rank_cache_empty_missing_and_existing(tmp_path):
    """路徑為空或不存在回傳空字典，存在時回傳內容。"""
    assert _load_rank_cache(None, "x") == {}
    assert _load_rank_cache(str(tmp_path / "none.json"), "x") == {}
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"a": {"expected_rank": 1.0}}), encoding="utf-8")
    assert _load_rank_cache(str(p), "x") == {"a": {"expected_rank": 1.0}}


def test_pipeline_requires_both_cand2_and_query2(tmp_path):
    """只指定其中一份 pub2/priv2 快取時應拒絕。"""
    with pytest.raises(ValueError):
        run_full_fused_pipeline(cand2_cache_path=str(tmp_path / "c.json"))
