"""
Runtime Model Self-Integrity Verification
==========================================
Probabilistically re-hashes neural network layer weights during inference
to detect in-memory tampering. Aborts execution upon integrity mismatch.
"""

import hashlib
import random
import logging
import torch
import torch.nn as nn

logger = logging.getLogger("RuntimeIntegrity")


class ModelTamperError(RuntimeError):
    """Raised when runtime integrity verification detects weight tampering."""
    pass


class LayerHashRegistry:
    """
    Maintains a SHA-256 hash of every named parameter at model load time.
    These reference hashes serve as the gold standard for integrity checks.
    """

    def __init__(self):
        self._hashes: dict[str, bytes] = {}

    def register_model(self, model: nn.Module):
        """Compute and store hashes for all named parameters."""
        self._hashes.clear()
        for name, param in model.named_parameters():
            param_hash = self._hash_tensor(param)
            self._hashes[name] = param_hash
        logger.info(f"Registered integrity hashes for {len(self._hashes)} parameters.")

    def get_registered_names(self) -> list[str]:
        return list(self._hashes.keys())

    def get_hash(self, name: str) -> bytes:
        return self._hashes[name]

    @staticmethod
    def _hash_tensor(t: torch.Tensor) -> bytes:
        """Compute SHA-256 of a tensor's raw bytes."""
        data = t.detach().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(data).digest()


class RuntimeIntegrityMonitor:
    """
    Probabilistically verifies model weight integrity during inference.
    
    At each check, a random subset of layers is re-hashed and compared
    against the reference registry. If any mismatch is found, the monitor
    immediately zeros all model weights and raises ModelTamperError.
    
    Args:
        model: The neural network to monitor
        registry: Precomputed hash registry from model load time
        check_probability: Fraction of layers to check per invocation (default 0.05 = 5%)
        check_interval: Number of inference steps between checks (default 1 = every step)
    """

    def __init__(
        self,
        model: nn.Module,
        registry: LayerHashRegistry,
        check_probability: float = 0.05,
        check_interval: int = 1,
    ):
        self.model = model
        self.registry = registry
        self.check_probability = check_probability
        self.check_interval = check_interval
        self._step_counter = 0
        self._checks_passed = 0
        self._total_checks = 0

    def step(self):
        """
        Called once per inference step. Probabilistically triggers an
        integrity check based on the configured interval and probability.
        """
        self._step_counter += 1
        if self._step_counter % self.check_interval != 0:
            return

        names = self.registry.get_registered_names()
        # Select a random subset of layers
        num_to_check = max(1, int(len(names) * self.check_probability))
        selected = random.sample(names, min(num_to_check, len(names)))

        for name in selected:
            self._total_checks += 1
            param = dict(self.model.named_parameters()).get(name)
            if param is None:
                self._emergency_wipe(f"Parameter '{name}' missing from model")
                return

            current_hash = LayerHashRegistry._hash_tensor(param)
            expected_hash = self.registry.get_hash(name)

            if current_hash != expected_hash:
                self._emergency_wipe(
                    f"TAMPER DETECTED in '{name}': "
                    f"expected {expected_hash[:8].hex()}, "
                    f"got {current_hash[:8].hex()}"
                )
                return

        self._checks_passed += num_to_check

    def get_stats(self) -> dict:
        return {
            "steps": self._step_counter,
            "total_checks": self._total_checks,
            "checks_passed": self._checks_passed,
        }

    def _emergency_wipe(self, reason: str):
        """Zero all model weights and raise a tamper error."""
        logger.critical(f"INTEGRITY VIOLATION: {reason}")
        logger.critical("EMERGENCY: Zeroing all model weights immediately.")

        for name, param in self.model.named_parameters():
            param.data.zero_()

        raise ModelTamperError(
            f"Model weights tampered. {reason}. "
            "All weights have been securely wiped. Inference aborted."
        )
