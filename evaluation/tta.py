"""
Test-Time Augmentation
======================
Runs inference over several views of each image and merges the detections.

Two augmentations are used, both chosen because their inverse is exact:

- **Horizontal flip.** Reflecting boxes back is lossless.
- **Multi-scale.** Rescaling boxes by 1/s is lossless up to interpolation of
  the image itself.

Rotations and shears are omitted deliberately: an axis-aligned box cannot
represent a rotated object, so inverting such a warp inflates every box.

Merged detections are resolved with class-wise NMS. TTA mainly helps small
and boundary-case objects, which is where this model is weakest, at the cost
of one forward pass per view.

``TTAModel`` exposes the same ``predict`` signature as ``SecureYOLO``, so it
can be handed to ``evaluate`` unchanged.
"""

import logging

import torch
import torch.nn.functional as F
import torchvision

logger = logging.getLogger("TTA")


def _round_to_stride(value, stride=32):
    """Round a spatial size up to a multiple of the network stride."""
    return int((value + stride - 1) // stride * stride)


class TTAModel:
    """Wraps a detector so ``predict`` aggregates several augmented views.

    Args:
        model: a ``SecureYOLO`` instance
        scales: input scale factors, relative to the incoming resolution
        flip: include a horizontally flipped view of each scale
        merge_iou: IoU threshold for the final merge NMS
    """

    def __init__(self, model, scales=(1.0,), flip=True, merge_iou=0.65, stride=32):
        self.model = model
        self.scales = tuple(scales)
        self.flip = flip
        self.merge_iou = merge_iou
        self.stride = stride
        self.num_classes = model.num_classes

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self

    @property
    def training(self):
        return self.model.training

    def parameters(self):
        return self.model.parameters()

    def _views(self, images):
        """Yield (tensor, scale, flipped) for each augmented view."""
        _, _, h, w = images.shape
        for s in self.scales:
            if s == 1.0:
                scaled = images
            else:
                new_h = _round_to_stride(h * s, self.stride)
                new_w = _round_to_stride(w * s, self.stride)
                scaled = F.interpolate(images, size=(new_h, new_w),
                                       mode="bilinear", align_corners=False)
            yield scaled, s, False
            if self.flip:
                yield scaled.flip(-1), s, True

    @torch.no_grad()
    def predict(self, images, conf_threshold=0.001, iou_threshold=0.7,
                max_detections=300, **kwargs):
        """Aggregate detections across views, in the input image's coordinates."""
        batch = images.shape[0]
        _, _, base_h, base_w = images.shape
        pooled = [[] for _ in range(batch)]

        for view, scale, flipped in self._views(images):
            view_w = view.shape[-1]
            dets = self.model.predict(
                view, conf_threshold=conf_threshold, iou_threshold=iou_threshold,
                max_detections=max_detections, **kwargs,
            )

            for bi, det in enumerate(dets):
                if det.numel() == 0:
                    continue
                det = det.clone()

                if flipped:
                    x1 = view_w - det[:, 2]
                    x2 = view_w - det[:, 0]
                    det[:, 0], det[:, 2] = x1, x2

                if scale != 1.0:
                    # Map back to the resolution the caller supplied.
                    det[:, [0, 2]] *= base_w / view_w
                    det[:, [1, 3]] *= base_h / view.shape[-2]

                pooled[bi].append(det)

        results = []
        for bi in range(batch):
            if not pooled[bi]:
                results.append(images.new_zeros((0, 6)))
                continue

            merged = torch.cat(pooled[bi], dim=0)
            keep = torchvision.ops.batched_nms(
                merged[:, :4], merged[:, 4], merged[:, 5].long(), self.merge_iou
            )[:max_detections]
            results.append(merged[keep])

        return results


def build_tta(model, mode="flip", merge_iou=0.65):
    """Construct a TTA wrapper from a named preset.

    Presets trade accuracy against the number of forward passes:

    - ``off``    : 1 pass, the plain model
    - ``flip``   : 2 passes, horizontal flip only
    - ``scale``  : 3 passes, multi-scale without flip
    - ``full``   : 6 passes, multi-scale with flip
    """
    presets = {
        "off": None,
        "flip": dict(scales=(1.0,), flip=True),
        "scale": dict(scales=(0.83, 1.0, 1.17), flip=False),
        "full": dict(scales=(0.83, 1.0, 1.17), flip=True),
    }
    if mode not in presets:
        raise ValueError(f"Unknown TTA mode {mode!r}. Choose from {list(presets)}")
    if presets[mode] is None:
        return model

    cfg = presets[mode]
    n_views = len(cfg["scales"]) * (2 if cfg["flip"] else 1)
    logger.info("TTA '%s': scales=%s flip=%s -> %d forward passes/image",
                mode, cfg["scales"], cfg["flip"], n_views)
    return TTAModel(model, merge_iou=merge_iou, **cfg)
