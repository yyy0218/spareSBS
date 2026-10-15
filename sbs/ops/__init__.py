"""Point operator API. Importing this package never compiles native code."""

import os


def configure_backend(name):
    if name not in {"native", "reference"}:
        raise ValueError(f"Unknown backend {name}")
    os.environ["SBS_BACKEND"] = name


def _backend():
    if os.getenv("SBS_BACKEND", "native") == "reference":
        from . import reference

        return reference
    from . import native

    return native


def directional_response(points, neighbors, weights, mode, training=True):
    backend = _backend()
    if backend.__name__.endswith("reference"):
        return backend.response(points, neighbors, weights, mode)
    function = {
        "spatial": backend.spatial_response,
        "rgb": backend.rgb_response,
        "normal": backend.normal_response,
    }[mode]
    return function(points, neighbors, weights, training)


def directional_pool(features, xyz, neighbors, weights, training=True):
    return _backend().directional_pool(features, xyz, neighbors, weights, training)


class KDTree:
    def __init__(self, xyz, max_leaf_size=20):
        self.impl = _backend().KDTree(xyz, max_leaf_size)

    def knn(self, query, k=1, ordered=False):
        return self.impl.knn(query, k, ordered)


def grid_subsampling(xyz, grid_size, hash_size=1.0):
    return _backend().grid_subsampling(xyz, grid_size, hash_size)


def grid_subsampling_test(xyz, grid_size, hash_size=1.0, pick=0):
    return _backend().grid_subsampling_test(xyz, grid_size, hash_size, pick)
