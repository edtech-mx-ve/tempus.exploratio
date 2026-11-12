"""Pruebas de persistencia e integridad con tmp_path."""
import importlib.util
import pytest
from tempus.demo import synthetic_series
from tempus.io import save_snapshot, load_snapshot, list_snapshots

@pytest.mark.skipif(importlib.util.find_spec("pyarrow") is None, reason="requiere pyarrow")
def test_roundtrip_and_versioning(tmp_path):
    dataset = synthetic_series()
    a = save_snapshot(dataset, "MSFT", tmp_path)
    b = save_snapshot(dataset, "MSFT", tmp_path)
    assert a != b
    assert len(list_snapshots("MSFT", tmp_path)) == 2
    output = load_snapshot(a, tmp_path, "MSFT")
    assert len(output) == len(dataset)
    with pytest.raises(ValueError, match="permitida"):
        load_snapshot(a, tmp_path, "XOM")
