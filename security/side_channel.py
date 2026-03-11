"""
Side-Channel Attack Mitigation
================================
Defenses against timing analysis, power analysis, and inference
fingerprinting attacks. Implements constant-time cryptographic
operations, CUDA scheduling randomization, and timing noise injection.
"""

import time
import hmac
import hashlib
import os
import random
import logging
import statistics
from typing import Optional, Callable

import torch

logger = logging.getLogger("SideChannel")


class ConstantTimeCrypto:
    """
    Wraps security-critical comparisons with constant-time equivalents
    to prevent timing side-channel leaks.
    """

    @staticmethod
    def compare(a: bytes, b: bytes) -> bool:
        """Constant-time comparison of two byte strings."""
        return hmac.compare_digest(a, b)

    @staticmethod
    def compare_hashes(hash_a: bytes, hash_b: bytes) -> bool:
        """Constant-time hash comparison."""
        return hmac.compare_digest(hash_a, hash_b)

    @staticmethod
    def secure_select(condition: bool, a: bytes, b: bytes) -> bytes:
        """
        Constant-time conditional select: returns a if condition else b.
        Both branches are always evaluated.
        """
        # Convert condition to a byte mask
        mask = bytes([0xFF if condition else 0x00] * max(len(a), len(b)))
        inv_mask = bytes([0x00 if condition else 0xFF] * max(len(a), len(b)))

        # Pad shorter input
        a_padded = a.ljust(len(mask), b'\x00')
        b_padded = b.ljust(len(mask), b'\x00')

        result = bytes(
            (a_byte & m) | (b_byte & im)
            for a_byte, b_byte, m, im in zip(a_padded, b_padded, mask, inv_mask)
        )
        return result[:max(len(a), len(b))]


class TimingNoiseInjector:
    """
    Injects calibrated random delays around inference calls to flatten
    the timing distribution and resist timing-based fingerprinting.
    """

    def __init__(self, target_latency_ms: float = 15.0, jitter_ms: float = 2.0):
        """
        Args:
            target_latency_ms: Target constant latency per inference (ms)
            jitter_ms: Random jitter range (±ms) added on top
        """
        self.target_latency_s = target_latency_ms / 1000.0
        self.jitter_s = jitter_ms / 1000.0

    def wrap_inference(self, inference_fn: Callable, *args, **kwargs):
        """
        Execute the inference function and pad the total time to the
        target latency plus random jitter.
        """
        start = time.perf_counter()
        result = inference_fn(*args, **kwargs)
        elapsed = time.perf_counter() - start

        # Calculate remaining time to hit target
        remaining = self.target_latency_s - elapsed
        jitter = random.uniform(-self.jitter_s, self.jitter_s)
        sleep_time = max(0, remaining + jitter)

        if sleep_time > 0:
            # Busy-wait for more precise timing than time.sleep()
            target = time.perf_counter() + sleep_time
            while time.perf_counter() < target:
                pass

        return result

    def measure_uniformity(self, latencies: list[float]) -> dict:
        """
        Analyze a list of inference latencies for timing uniformity.
        
        Returns:
            dict with mean, std, cv (coefficient of variation), min, max
        """
        if len(latencies) < 2:
            return {"error": "Need at least 2 samples"}

        mean = statistics.mean(latencies)
        std = statistics.stdev(latencies)
        cv = std / mean if mean > 0 else float('inf')

        return {
            "mean_ms": mean * 1000,
            "std_ms": std * 1000,
            "cv": cv,
            "cv_percent": cv * 100,
            "min_ms": min(latencies) * 1000,
            "max_ms": max(latencies) * 1000,
            "uniformity_pass": cv < 0.05,  # < 5% CV target
        }


class CUDAScheduleRandomizer:
    """
    Randomizes CUDA stream execution order and inserts dummy kernel
    launches to mask real inference patterns from power/EM analysis.
    """

    def __init__(self, num_dummy_streams: int = 2, enabled: bool = True):
        self.num_dummy_streams = num_dummy_streams
        self.enabled = enabled and torch.cuda.is_available()
        self._dummy_streams: list = []
        self._dummy_tensors: list = []

        if self.enabled:
            for _ in range(num_dummy_streams):
                self._dummy_streams.append(torch.cuda.Stream())
            logger.info(f"CUDA schedule randomizer active with {num_dummy_streams} dummy streams.")

    def pre_inference_noise(self):
        """Launch dummy CUDA operations before real inference."""
        if not self.enabled:
            return

        # Launch dummy kernels on separate streams
        for stream in self._dummy_streams:
            with torch.cuda.stream(stream):
                # Random matrix multiply (dummy work)
                size = random.randint(32, 128)
                a = torch.randn(size, size, device='cuda')
                b = torch.randn(size, size, device='cuda')
                c = torch.mm(a, b)
                # Immediately zero and discard
                c.zero_()
                a.zero_()
                b.zero_()
                del a, b, c

    def post_inference_noise(self):
        """Launch dummy CUDA operations after real inference."""
        if not self.enabled:
            return

        # Random selection of which streams get dummy work
        active_streams = random.sample(
            self._dummy_streams,
            k=random.randint(1, len(self._dummy_streams))
        )
        for stream in active_streams:
            with torch.cuda.stream(stream):
                size = random.randint(16, 64)
                t = torch.randn(size, size, device='cuda')
                _ = torch.relu(t)
                t.zero_()
                del t

    def synchronize(self):
        """Wait for all streams including dummy ones."""
        if self.enabled:
            torch.cuda.synchronize()


class LatencyBenchmark:
    """Records and analyzes inference latency for side-channel assessment."""

    def __init__(self):
        self._latencies: list[float] = []

    def record(self, latency_seconds: float):
        self._latencies.append(latency_seconds)

    def analyze(self) -> dict:
        """Produce a full timing uniformity report."""
        injector = TimingNoiseInjector()
        return injector.measure_uniformity(self._latencies)

    def reset(self):
        self._latencies.clear()

    @property
    def count(self) -> int:
        return len(self._latencies)
