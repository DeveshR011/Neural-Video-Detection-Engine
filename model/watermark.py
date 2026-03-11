"""
Neural Model Watermarking for IP Protection
=============================================
Embeds and verifies cryptographic watermark signatures in the model's
weight space. Designed to survive fine-tuning, pruning, and quantization.
"""

import hashlib
import hmac
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger("Watermark")


class WatermarkEmbedder:
    """
    Injects a cryptographic signature into selected layers' weight matrices
    using a secret-key-derived pseudo-random mask. The inner product of
    (weights · mask) encodes a digital signature bit-string.
    
    The watermark is spread across many layers to resist removal attacks.
    """

    def __init__(self, secret_key: bytes, signature: str = "NEURAL_ENGINE_v1"):
        self.secret_key = secret_key
        self.signature = signature
        self._signature_bits = self._string_to_bits(signature)

    def embed(self, model: nn.Module, strength: float = 0.001) -> nn.Module:
        """
        Embed the watermark signature into the model weights.
        
        Args:
            model: The neural network to watermark
            strength: Perturbation strength (lower = less impact on accuracy)
        
        Returns:
            The watermarked model (modified in-place)
        """
        eligible_layers = self._get_eligible_layers(model)
        if not eligible_layers:
            logger.warning("No eligible layers for watermarking.")
            return model

        bits = self._signature_bits
        bits_per_layer = max(1, len(bits) // len(eligible_layers))

        for i, (name, param) in enumerate(eligible_layers):
            start_bit = i * bits_per_layer
            end_bit = min(start_bit + bits_per_layer, len(bits))
            layer_bits = bits[start_bit:end_bit]

            if not layer_bits:
                continue

            # Generate deterministic pseudo-random mask from secret + layer name
            mask = self._generate_mask(name, param.shape, len(layer_bits))
            mask = mask.to(param.device)

            # Embed each bit by adjusting the dot product direction
            with torch.no_grad():
                flat_weights = param.data.flatten()
                flat_mask = mask.flatten()[:flat_weights.numel()]

                for j, bit in enumerate(layer_bits):
                    # Each bit uses a subsection of the mask
                    segment_size = flat_weights.numel() // len(layer_bits)
                    start = j * segment_size
                    end = start + segment_size

                    segment_mask = flat_mask[start:end]
                    current_dot = torch.dot(flat_weights[start:end], segment_mask).item()

                    target_sign = 1.0 if bit == 1 else -1.0
                    if (current_dot > 0) != (target_sign > 0):
                        # Nudge weights to flip the dot product sign
                        flat_weights[start:end] += target_sign * strength * segment_mask

                param.data = flat_weights.reshape(param.shape)

        logger.info(
            f"Watermark embedded: {len(bits)} bits across "
            f"{len(eligible_layers)} layers (strength={strength})"
        )
        return model

    def _get_eligible_layers(self, model: nn.Module) -> list[tuple[str, nn.Parameter]]:
        """Select layers suitable for watermarking (large weight matrices)."""
        eligible = []
        for name, param in model.named_parameters():
            if "weight" in name and param.numel() > 1000:
                eligible.append((name, param))
        return eligible

    def _generate_mask(self, layer_name: str, shape: tuple, num_bits: int) -> torch.Tensor:
        """Generate a deterministic pseudo-random mask from the secret key and layer name."""
        seed_bytes = hmac.new(
            self.secret_key, layer_name.encode("utf-8"), hashlib.sha256
        ).digest()
        seed = int.from_bytes(seed_bytes[:4], "big")
        rng = np.random.RandomState(seed)
        mask_np = rng.randn(*shape).astype(np.float32)
        return torch.from_numpy(mask_np)

    @staticmethod
    def _string_to_bits(s: str) -> list[int]:
        """Convert a string to a list of bits."""
        bits = []
        for byte in s.encode("utf-8"):
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return bits


class WatermarkVerifier:
    """
    Extracts and verifies a watermark from a suspect model copy.
    """

    def __init__(self, secret_key: bytes, signature: str = "NEURAL_ENGINE_v1"):
        self.embedder = WatermarkEmbedder(secret_key, signature)
        self._expected_bits = self.embedder._signature_bits

    def verify(self, model: nn.Module, threshold: float = 0.7) -> tuple[bool, float]:
        """
        Verify whether the model contains the expected watermark.
        
        Args:
            model: The suspect model to verify
            threshold: Minimum fraction of correctly extracted bits
            
        Returns:
            (is_watermarked, bit_accuracy)
        """
        eligible_layers = self.embedder._get_eligible_layers(model)
        if not eligible_layers:
            return False, 0.0

        bits = self._expected_bits
        bits_per_layer = max(1, len(bits) // len(eligible_layers))
        extracted_bits: list[int] = []

        for i, (name, param) in enumerate(eligible_layers):
            start_bit = i * bits_per_layer
            end_bit = min(start_bit + bits_per_layer, len(bits))
            layer_bits = bits[start_bit:end_bit]

            if not layer_bits:
                continue

            mask = self.embedder._generate_mask(name, param.shape, len(layer_bits))
            mask = mask.to(param.device)

            with torch.no_grad():
                flat_weights = param.data.flatten()
                flat_mask = mask.flatten()[:flat_weights.numel()]

                for j in range(len(layer_bits)):
                    segment_size = flat_weights.numel() // len(layer_bits)
                    start = j * segment_size
                    end = start + segment_size

                    dot = torch.dot(flat_weights[start:end], flat_mask[start:end]).item()
                    extracted_bits.append(1 if dot > 0 else 0)

        # Compare extracted bits to expected
        if not extracted_bits:
            return False, 0.0

        correct = sum(
            e == x for e, x in zip(self._expected_bits[:len(extracted_bits)], extracted_bits)
        )
        accuracy = correct / len(extracted_bits)

        is_watermarked = accuracy >= threshold
        logger.info(
            f"Watermark verification: accuracy={accuracy:.2%} "
            f"({'PASS' if is_watermarked else 'FAIL'}, threshold={threshold:.0%})"
        )
        return is_watermarked, accuracy
