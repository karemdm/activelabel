import numpy as np

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
