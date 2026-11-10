"""Small, independently testable helpers for segment-normalized CoT SFT."""

import math

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def cosine_cot_weight(step, total_steps, start=1.0, end=0.5):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    progress = min(max(step / total_steps, 0.0), 1.0)
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))


def segment_normalized_loss(logits, labels, segments, cot_weight, answer_weight=1.0):
    """Causal shift; mean per segment per sample, then mean across samples.

    Segment IDs: 0 = ignored, 1 = reasoning, 2 = answer/termination.
    Checkpoint small CE chunks to avoid retaining full-vocabulary FP32 copies.
    """
    shifted_labels = labels[:, 1:]
    shifted_segments = segments[:, 1:]
    sample_losses = []
    for row in range(labels.shape[0]):
        losses = []
        for segment, weight in ((1, cot_weight), (2, answer_weight)):
            mask = shifted_labels[row].ne(-100) & shifted_segments[row].eq(segment)
            positions = mask.nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                raise ValueError(f"Sample {row} has no supervised tokens for segment {segment}")
            total = logits.new_zeros((), dtype=torch.float32)
            for chunk in positions.split(64):
                selected = logits[row, chunk, :]
                targets = shifted_labels[row, chunk]
                def chunk_ce(values, gold):
                    return F.cross_entropy(values.float(), gold, reduction="sum")
                if torch.is_grad_enabled() and selected.requires_grad:
                    total = total + checkpoint(chunk_ce, selected, targets, use_reentrant=False)
                else:
                    total = total + chunk_ce(selected, targets)
            losses.append(weight * total / positions.numel())
        sample_losses.append(sum(losses))
    return torch.stack(sample_losses).mean()
