import numpy as np
import pytest

from activelabel.curation import similarity


def blobs(sizes, dims=16, seed=0):
    """Rows around one far-apart centre per group, with the true group of each row."""
    rng = np.random.default_rng(seed)
    rows, truth = [], []
    for group, size in enumerate(sizes):
        centre = np.zeros(dims)
        centre[group] = 10.0
        rows.append(centre + rng.normal(0, 0.1, (size, dims)))
        truth += [group] * size
    return np.vstack(rows), np.array(truth)


def test_the_groups_follow_the_blobs_largest_first():
    vectors, truth = blobs([12, 8, 5])
    labels = similarity.find_groups(vectors)
    assert sorted(set(labels.tolist())) == [0, 1, 2]
    for group in range(3):  # each blob is one group, and group 0 is the largest
        assert len(set(labels[truth == group].tolist())) == 1
    assert labels[truth == 0][0] == 0 and labels[truth == 2][0] == 2


def test_tiny_or_identical_sets_are_one_group():
    assert similarity.find_groups(np.ones((3, 4))).tolist() == [0, 0, 0]
    assert similarity.find_groups(np.ones((10, 4))).tolist() == [0] * 10


def test_points_are_scaled_to_the_unit_square_with_the_method_by_size():
    vectors, _ = blobs([6, 6])
    points, method = similarity.project_points(vectors)
    assert method == "tsne" and points.shape == (12, 2)
    assert points.min() >= 0 and points.max() <= 1 and np.isclose(points.max(axis=0), 1).all()
    assert similarity.project_points(vectors[:3])[1] == "pca"
    points, method = similarity.project_points(vectors[:1])
    assert method == "none" and points.tolist() == [[0.5, 0.5]]


def test_pca_of_one_dimension_gets_a_flat_second_axis():
    points, method = similarity.project_points(np.array([[0.0], [1.0], [3.0]]))
    assert method == "pca" and points.shape == (3, 2) and points[:, 1].tolist() == [0, 0, 0]


def test_build_map_names_the_typical_image_of_every_group():
    vectors, truth = blobs([5, 4])
    ids = [f"i{k}" for k in range(len(vectors))]
    result = similarity.build_map(ids, vectors)
    assert [g["size"] for g in result["groups"]] == [5, 4]
    assert result["groups"][0]["typical"] in ids[:5] and result["groups"][1]["typical"] in ids[5:]
    assert result["points"]["ids"] == ids and len(result["points"]["xy"]) == 9
    assert result["points"]["group"] == [0] * 5 + [1] * 4


def test_store_keeps_the_state_and_the_vectors(tmp_path):
    store = similarity.SimilarityStore(tmp_path)
    assert store.read() == {} and store.load_vectors() is None
    store.update(status="queued")
    assert store.update(done=2) == {"status": "queued", "done": 2}
    store.save_vectors(["a", "b"], np.eye(2), "resnet18")
    ids, vectors, backbone = store.load_vectors()
    assert ids == ["a", "b"] and vectors.dtype == np.float32 and backbone == "resnet18"
    store.clear()
    assert not store.state_path.exists() and not store.vectors_path.exists()


def test_a_map_is_outdated_when_the_active_images_change():
    state = {"points": {"ids": ["a", "b"]}, "unreadable": ["c"]}
    assert not similarity.is_outdated(state, {"a", "b", "c"})  # an unreadable image is not missing
    assert similarity.is_outdated(state, {"a", "b", "c", "d"})
    assert similarity.is_outdated(state, {"a"})
    assert not similarity.is_outdated({}, {"a"})


def test_cluster_auto_is_find_groups():
    vectors, _ = blobs([12, 8, 5])
    assert similarity.cluster(vectors).tolist() == similarity.find_groups(vectors).tolist()
    assert similarity.cluster(vectors, "auto", 7).tolist() == similarity.find_groups(vectors).tolist()


def test_cluster_kmeans_gives_the_asked_number_largest_first():
    vectors, truth = blobs([12, 8, 5, 3])
    labels = similarity.cluster(vectors, "kmeans", groups=4)
    assert sorted(set(labels.tolist())) == [0, 1, 2, 3]
    assert [int((labels == g).sum()) for g in range(4)] == [12, 8, 5, 3]
    assert len(set(labels[truth == 1].tolist())) == 1
    assert len(set(similarity.cluster(vectors, "kmeans", groups=2).tolist())) == 2


def test_cluster_hdbscan_finds_blobs_and_marks_an_outlier_as_noise():
    vectors, truth = blobs([12, 9])
    outlier = np.full((1, vectors.shape[1]), 50.0)
    labels = similarity.cluster(np.vstack([vectors, outlier]), "hdbscan", min_size=4)
    assert labels[-1] == similarity.NOISE
    assert sorted(set(labels[:-1].tolist())) == [0, 1]
    assert len(set(labels[:12].tolist())) == 1 and labels[0] == 0  # the largest blob is group 0


def test_cluster_hdbscan_with_nothing_dense_is_all_noise():
    vectors = np.random.default_rng(3).normal(size=(10, 16)) * np.arange(1, 11)[:, None] ** 2
    labels = similarity.cluster(vectors, "hdbscan", min_size=9)
    assert set(labels.tolist()) <= {similarity.NOISE, 0}


@pytest.mark.parametrize(
    ("method", "groups", "min_size"),
    [("dbscan", 3, 5), ("kmeans", 1, 5), ("kmeans", similarity.MAX_CHOSEN_GROUPS + 1, 5), ("hdbscan", 0, 1)],
)
def test_cluster_refuses_a_method_or_number_out_of_range(method, groups, min_size):
    vectors, _ = blobs([12, 8])
    with pytest.raises(ValueError):
        similarity.cluster(vectors, method, groups, min_size)


def test_cluster_with_fewer_than_four_images_is_one_group():
    assert similarity.cluster(np.eye(3), "kmeans", groups=2).tolist() == [0, 0, 0]
    assert similarity.cluster(np.eye(3), "hdbscan").tolist() == [0, 0, 0]
    assert similarity.cluster(np.ones((10, 4)), "hdbscan").tolist() == [0] * 10


def test_typical_images_ignores_the_noise():
    vectors, _ = blobs([5, 4])
    ids = [f"i{k}" for k in range(len(vectors))]
    labels = np.array([0] * 5 + [1] * 3 + [similarity.NOISE])
    groups = similarity.typical_images(ids, vectors, labels)
    assert [(g["group"], g["size"]) for g in groups] == [(0, 5), (1, 3)]
    assert all(g["typical"] != "i8" for g in groups)


def test_near_pairs_finds_almost_equal_embeddings_once():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(4, 32))
    vectors = np.vstack([base, base[0] + 1e-3 * rng.normal(size=32), base[1] * 3.0])  # a copy, a rescaled one
    ids = ["a", "b", "c", "d", "a2", "b3"]
    pairs = similarity.near_pairs(ids, vectors)
    assert {(p[0], p[1]) for p in pairs} == {("a", "a2"), ("b", "b3")}
    assert all(0 <= p[2] <= similarity.PAIR_LIMIT for p in pairs)
    assert pairs == sorted(pairs, key=lambda p: p[2])


def test_near_pairs_respects_the_limit_and_small_sets():
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    assert similarity.near_pairs(["x"], vectors[:1]) == []
    assert similarity.near_pairs(["x", "y", "z"], vectors, limit=0.0) == []
    close = similarity.near_pairs(["x", "y", "z"], vectors, limit=0.1)
    assert [p[:2] for p in close] == [["x", "y"]]


def test_near_pairs_in_blocks_gives_the_same_pairs(monkeypatch):
    rng = np.random.default_rng(2)
    base = rng.normal(size=(10, 8))
    vectors = np.vstack([base, base + 1e-4])
    ids = [f"i{n}" for n in range(20)]
    whole = similarity.near_pairs(ids, vectors)
    monkeypatch.setattr(similarity, "PAIR_BLOCK", 20)  # one row at a time
    assert similarity.near_pairs(ids, vectors) == whole
    assert len(whole) == 10


def test_build_map_keeps_the_near_pairs():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(6, 16))
    vectors = np.vstack([base, base[:1]])
    result = similarity.build_map([f"i{n}" for n in range(7)], vectors)
    assert [p[:2] for p in result["near_pairs"]] == [["i0", "i6"]]
