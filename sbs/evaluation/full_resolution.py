"""Full-resolution single-scan/room voting for the two indoor benchmarks."""

import torch
from tqdm import tqdm
from sbs.engine.runtime import precision_context
from .metrics import update_confusion, summarize


@torch.no_grad()
def evaluate(model, dataset, cfg, device, max_scenes=0):
    votes = cfg["evaluation"]["votes"]
    classes = cfg["model"]["num_classes"]
    if len(dataset) % votes:
        raise ValueError("Dataset length does not match vote count")
    scenes = len(dataset) // votes
    if max_scenes:
        scenes = min(scenes, max_scenes)
    hist = torch.zeros((classes, classes), dtype=torch.int64)
    model.eval()
    for scene in tqdm(range(scenes), desc=f"Full resolution / {votes} views"):
        probabilities = None
        target = None
        for view in range(votes):
            xyz, features, indices, back, labels = dataset[scene * votes + view]
            indices = [index.to(device).long() for index in indices[::-1]]
            with precision_context(device, cfg["training"]["precision"]):
                logits = model.predict(xyz.to(device), features.to(device), indices)
                source = logits
                full = source.softmax(-1)[back.to(device).long()].float()
            probabilities = full if probabilities is None else probabilities + full
            target = labels.to(device).long()
        update_confusion(hist, probabilities.argmax(-1), target, cfg["ignore_index"])
    result = summarize(hist)
    result.update(num_scenes=scenes, votes=votes, protocol=cfg["evaluation"]["protocol"])
    return result
