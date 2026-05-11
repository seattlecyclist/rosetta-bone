import numpy as np

from rosetta_bone.storyteller.retrieval.embed import Embedder


def test_embed_returns_normalized_vectors():
    e = Embedder("BAAI/bge-small-en-v1.5")
    vecs = e.embed(["hello", "world"])
    assert vecs.shape == (2, 384)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)
