"""Sparse-anchor preservation policies used by the main experiments."""


def normalize_anchor_retention(mode):
    mode = "preserve" if mode is None else str(mode).strip().lower()
    if mode not in {"preserve", "local_crop"}:
        raise ValueError("Only original and local_crop are main-experiment policies")
    return mode
