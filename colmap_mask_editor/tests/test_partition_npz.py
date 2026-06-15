"""V0.9: partition.npz の build/save/load/verify (allow_pickle=False, 原子保存)。"""

import numpy as np
import pytest

from ai import partition_npz
from ai.partition_npz import PartitionNpzError, REQUIRED_ARRAYS

from tests._partition_helpers import simple_three_leaf


def test_build_and_verify():
    arrays = simple_three_leaf()
    # build 直後にメモリ上で検証はできないので保存して読む
    assert int(arrays["leaf_count"][0]) == 3
    assert int(arrays["node_count"][0]) == 5
    assert int(arrays["root_id"][0]) == 5


def test_save_load_roundtrip(tmp_path):
    arrays = simple_three_leaf()
    path = tmp_path / "partition.npz"
    sha = partition_npz.save_partition_npz(path, arrays)
    assert len(sha) == 64
    data = partition_npz.load_partition_npz(path)
    assert int(data["leaf_count"][0]) == 3


def test_loads_with_allow_pickle_false(tmp_path):
    arrays = simple_three_leaf()
    path = tmp_path / "partition.npz"
    partition_npz.save_partition_npz(path, arrays)
    with np.load(path, allow_pickle=False) as data:
        assert "run_region_ids" in data.files


def test_dtypes_match_schema(tmp_path):
    arrays = simple_three_leaf()
    path = tmp_path / "partition.npz"
    partition_npz.save_partition_npz(path, arrays)
    data = partition_npz.load_partition_npz(path)
    for name, dtype in REQUIRED_ARRAYS.items():
        assert data[name].dtype == dtype, name


def test_no_dense_label_map(tmp_path):
    arrays = simple_three_leaf()
    path = tmp_path / "partition.npz"
    partition_npz.save_partition_npz(path, arrays)
    data = partition_npz.load_partition_npz(path)
    # H*W の dense 配列が無いこと: すべて run-length 由来で小さい
    for name, a in data.items():
        assert a.size < 100, f"{name} が大きすぎます ({a.size})"


def test_verify_detects_corrupt(tmp_path):
    arrays = dict(simple_three_leaf())
    # node_count を壊す
    arrays["node_count"] = np.asarray([99], dtype=np.uint32)
    path = tmp_path / "bad.npz"
    with pytest.raises(PartitionNpzError):
        # save は内部 verify で落ちる
        partition_npz.save_partition_npz(path, arrays)
    assert not path.exists()
    assert not (tmp_path / "bad.npz.tmp").exists()


def test_verify_detects_area_mismatch(tmp_path):
    arrays = dict(simple_three_leaf())
    area = arrays["node_area"].copy()
    area[4] = 999  # root の面積を壊す
    arrays["node_area"] = area
    path = tmp_path / "bad2.npz"
    with pytest.raises(PartitionNpzError):
        partition_npz.save_partition_npz(path, arrays)


def test_node_index_is_id_minus_one():
    arrays = simple_three_leaf()
    # root_id=5, parent[root-1]=0
    assert int(arrays["node_parent"][5 - 1]) == 0
    # 葉 1,2 の parent は node 4
    assert int(arrays["node_parent"][0]) == 4
    assert int(arrays["node_parent"][1]) == 4


def test_leaf_nodes_have_no_children():
    arrays = simple_three_leaf()
    for leaf in (1, 2, 3):
        assert int(arrays["node_left"][leaf - 1]) == 0
        assert int(arrays["node_right"][leaf - 1]) == 0


def test_sha256_stable(tmp_path):
    arrays = simple_three_leaf()
    p1 = tmp_path / "a.npz"
    p2 = tmp_path / "b.npz"
    s1 = partition_npz.save_partition_npz(p1, arrays)
    s2 = partition_npz.save_partition_npz(p2, simple_three_leaf())
    assert s1 == partition_npz.file_sha256(p1)
    # 同一内容は同一 SHA (savez_compressed は決定的)
    assert s1 == s2
