import torch

def mean_iou(preds, targets, num_classes):
    """
    Computes mean IoU for per-point segmentation.
    preds, targets: shape (B,N)
    """
    ious = []
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    for cls in range(num_classes):
        pred_cls = preds == cls
        target_cls = targets == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return sum(ious) / len(ious)
