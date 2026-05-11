from pathlib import Path

import numpy as np

from rosetta_bone.storyteller.retrieval.index import PillarIndex


def test_build_save_query_round_trip(tmp_path: Path):
    rng = np.random.default_rng(0)
    vecs = rng.standard_normal((10, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [f"id{i}" for i in range(10)]

    idx = PillarIndex.build(vecs, ids)
    p = tmp_path / "x.faiss"
    idx.save(p)

    loaded = PillarIndex.load(p, ids)
    sims, hits = loaded.query(vecs[3], top_k=3)
    assert hits[0] == "id3"
    assert sims[0] > 0.99
