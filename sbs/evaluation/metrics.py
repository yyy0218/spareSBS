"""Point-level confusion matrix and class-averaged segmentation metrics."""

import torch


def update_confusion(matrix, prediction, target, ignore_index):
    classes = matrix.shape[0]
    target = target.long()
    prediction = prediction.long()
    mask = (target != ignore_index) & (target >= 0) & (target < classes)
    encoded = target[mask] * classes + prediction[mask]
    matrix += (
        torch.bincount(encoded, minlength=classes * classes)
        .reshape(classes, classes)
        .to(matrix.device)
    )


def summarize(matrix):
    h = matrix.float()
    intersection = h.diag()
    count = h.sum(1)
    union = count + h.sum(0) - intersection
    iou = intersection / union.clamp_min(1)
    accuracy = intersection / count.clamp_min(1)
    valid = union > 0
    return {
        "oa": float(intersection.sum() / h.sum().clamp_min(1)),
        "macc": float(accuracy[count > 0].mean()) if (count > 0).any() else 0.0,
        "miou": float(iou[valid].mean()) if valid.any() else 0.0,
        "miou_all": float(iou.mean()),
        "per_class_iou": iou.tolist(),
        "num_valid_points": int(matrix.sum()),
        "confusion_matrix": matrix.tolist(),
    }
