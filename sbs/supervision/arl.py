"""Anchor Relay Learning. This module is imported by the trainer, not the model."""

from torch.nn import functional as F
from .local_anchor_relay import LocalAnchorRelay
from .global_semantic_confirmation import GlobalSemanticConfirmation
from .support import boundary_risk


def relay_loss(logits, targets):
    if targets is None or not targets.mask.any():
        return logits.sum() * 0
    mask = targets.mask
    weight = targets.weights[mask].float().clamp_min(1e-4)
    ce = F.cross_entropy(logits[mask].float(), targets.labels[mask], reduction="none")
    return (ce * weight).sum() / weight.sum().clamp_min(1e-4)


def local_weight_for_epoch(config: dict, epoch: int) -> float:
    """Local ARL coefficient for a zero-based training epoch.

    A five-epoch warm-in starts at one fifth of the target on start_epoch,
    reaches the full target at start_epoch + 4, and stays there. The ramp
    weights the supervision loss; it does not change selected relay targets.
    Omitting ramp_epochs preserves immediate activation for other recipes.
    """
    ramp_epochs = config.get("ramp_epochs", 0)
    if isinstance(ramp_epochs, bool) or not isinstance(ramp_epochs, int) or ramp_epochs < 0:
        raise ValueError("ARL local ramp_epochs must be a non-negative integer")
    start = config["start_epoch"]
    if epoch < start:
        return 0.0
    target = float(config["loss_weight"])
    if ramp_epochs == 0:
        return target
    progress = min(1.0, float(epoch - start + 1) / float(ramp_epochs))
    return target * max(0.0, progress)


class AnchorRelayLearning:
    def __init__(self, config, num_classes, ignore_index):
        self.config = config
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.local = LocalAnchorRelay(config["local"])
        self.global_confirmation = GlobalSemanticConfirmation(config["global_confirmation"])

    def local_weight(self, epoch: int) -> float:
        return local_weight_for_epoch(self.config["local"], epoch)

    def __call__(self, output, labels, epoch):
        cfg = self.config
        logits = output["logits"]
        local = global_targets = None
        local_loss = global_loss = logits.sum() * 0
        if epoch >= cfg["local"]["start_epoch"]:
            boundary = boundary_risk(output["support"])
            local = self.local(
                labels, output["support"], boundary, self.num_classes, self.ignore_index
            )
            local_loss = relay_loss(logits, local)
            if epoch >= cfg["global_confirmation"]["start_epoch"]:
                global_targets = self.global_confirmation(
                    logits,
                    output["semantic_features"],
                    labels,
                    local,
                    boundary,
                    self.num_classes,
                    self.ignore_index,
                )
                global_loss = relay_loss(logits, global_targets)
        local_weight = self.local_weight(epoch)
        loss = local_weight * local_loss + cfg["global_confirmation"]["loss_weight"] * global_loss
        metrics = {
            "local_loss_weight": local_weight,
            "local_loss": float(local_loss.detach()),
            "global_loss": float(global_loss.detach()),
            "local_points": 0 if local is None else int(local.mask.sum()),
            "global_points": 0 if global_targets is None else int(global_targets.mask.sum()),
        }
        return loss, metrics
