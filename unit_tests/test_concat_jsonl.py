import pytest

from aicup.concat_jsonl import concat


def test_concat_order_and_missing_newline(tmp_path):
    a, b, out = tmp_path / 'a.jsonl', tmp_path / 'b.jsonl', tmp_path / 'out.jsonl'
    a.write_bytes(b'{"key": "D_1"}\n{"key": "D_2"}')
    b.write_bytes(b'{"key": "A_1"}\n\n')
    assert concat([str(a), str(b)], str(out)) == 3
    assert out.read_bytes() == b'{"key": "D_1"}\n{"key": "D_2"}\n{"key": "A_1"}\n'


def test_concat_rejects_duplicate_key(tmp_path):
    a, out = tmp_path / 'a.jsonl', tmp_path / 'out.jsonl'
    a.write_bytes(b'{"key": "D_1"}\n')
    with pytest.raises(ValueError):
        concat([str(a), str(a)], str(out))
    assert not out.exists() and not (tmp_path / 'out.jsonl.part').exists()
