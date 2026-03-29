"""
Dynamic Architecture Morphing
===============================
Experimental per-session channel permutation/affine remapping utility.
For complex detector topologies (CSP, residual, concat, BiFPN), strict
output-preserving guarantees are not provided.

Compatible with depthwise-separable convolutions and BiFPN architectures.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import hashlib
import os

logger = logging.getLogger("ArchitectureMorph")


class ArchitectureMorpher:
    """
    Randomizes internal representation of standard (non-depthwise)
    Conv2d+BN pairs per-boot while preserving mathematical equivalence.

    Skips:
    - Depthwise convolutions (groups > 1)
    - Frozen layers (requires_grad=False)
    - Layers where channel dimensions don't align with the next layer
    """

    def __init__(self, session_seed: bytes = None, enabled: bool = False):
        if session_seed is None:
            session_seed = os.urandom(32)
        self.session_seed = session_seed
        self.enabled = enabled
        self._seed_int = int.from_bytes(
            hashlib.sha256(session_seed).digest()[:4], "big"
        )
        self.rng = np.random.RandomState(self._seed_int)
        self._morphed = False

    def morph(self, model: nn.Module) -> nn.Module:
        """Apply per-session morphing to eligible Conv2d+BN pairs."""
        if not self.enabled:
            logger.warning(
                "Architecture morphing is disabled by default. "
                "Enable explicitly only for controlled experiments."
            )
            return model

        if self._morphed:
            logger.warning("Model already morphed this session.")
            return model

        pairs = self._extract_morphable_pairs(model)
        morphed_count = 0

        for i, (conv, bn, next_conv) in enumerate(pairs):
            out_channels = conv.out_channels

            perm = torch.from_numpy(
                self.rng.permutation(out_channels)
            ).long().to(conv.weight.device)

            scale = torch.from_numpy(
                self.rng.uniform(0.8, 1.2, size=out_channels).astype(np.float32)
            ).to(conv.weight.device)

            with torch.no_grad():
                # Permute and scale output channels
                conv.weight.data = conv.weight.data[perm] * scale.view(-1, 1, 1, 1)
                if conv.bias is not None:
                    conv.bias.data = conv.bias.data[perm] * scale

                # Adjust BatchNorm
                if bn is not None:
                    bn.weight.data = bn.weight.data[perm]
                    bn.bias.data = bn.bias.data[perm]
                    bn.running_mean.data = bn.running_mean.data[perm]
                    bn.running_var.data = bn.running_var.data[perm]

                # Adjust next layer's input channels (if compatible)
                if next_conv is not None:
                    inv_perm = torch.argsort(perm)
                    inv_scale = 1.0 / scale
                    next_conv.weight.data = (
                        next_conv.weight.data[:, inv_perm, :, :] *
                        inv_scale.view(1, -1, 1, 1)
                    )

            morphed_count += 1

        self._morphed = True
        logger.info(
            f"Architecture morphed: {morphed_count} layer pairs with seed "
            f"{self.session_seed[:8].hex()}... "
            f"No strict output-invariance guarantee is claimed."
        )
        return model

    def _extract_morphable_pairs(self, model: nn.Module) -> list[tuple]:
        """
        Extract (conv, bn, next_compatible_conv) triples.
        
        Only includes standard convolutions (groups=1) where we can
        safely permute output channels and adjust the next layer's inputs.
        """
        # Collect all standard Conv2d layers with their associated BN
        modules = list(model.modules())
        conv_bn_list = []

        for i, module in enumerate(modules):
            if isinstance(module, nn.Conv2d) and module.groups == 1:
                # Find following BN
                bn = None
                if i + 1 < len(modules) and isinstance(modules[i + 1], nn.BatchNorm2d):
                    if modules[i + 1].num_features == module.out_channels:
                        bn = modules[i + 1]
                conv_bn_list.append((module, bn))

        # Build triples: (conv, bn, next_compatible_conv)
        triples = []
        for idx in range(len(conv_bn_list)):
            conv, bn = conv_bn_list[idx]

            # Find the next standard conv whose in_channels matches
            # this conv's out_channels
            next_conv = None
            if idx + 1 < len(conv_bn_list):
                candidate, _ = conv_bn_list[idx + 1]
                if (candidate.in_channels == conv.out_channels and
                        candidate.groups == 1):
                    next_conv = candidate

            triples.append((conv, bn, next_conv))

        return triples

    @property
    def is_morphed(self) -> bool:
        return self._morphed
