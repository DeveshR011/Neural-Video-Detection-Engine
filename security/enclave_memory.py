"""
Enclave-Bound Tensor Memory Allocator
======================================
Custom allocator that forces PyTorch tensor storage into locked,
enclave-bound memory pages. On release, performs a 3-pass overwrite
(zeros → random → zeros) before unlocking.
"""

import ctypes
import os
import sys
import logging
import torch

logger = logging.getLogger("EnclaveMemory")


class EnclaveAllocator:
    """
    Allocates and manages memory regions that are:
    1. Locked into physical RAM (no swap)
    2. Wiped securely on deallocation
    3. Tracked for enclave lifecycle management
    
    Usage:
        allocator = EnclaveAllocator()
        tensor = allocator.allocate_secure_tensor((3, 640, 640), dtype=torch.float32)
        # ... use tensor ...
        allocator.release_tensor(tensor)
    """

    def __init__(self):
        self._allocations: dict[int, tuple[ctypes.Array, int]] = {}
        self._tensor_map: dict[int, int] = {}  # tensor id → allocation address

        if os.name == 'nt':
            self._kernel32 = ctypes.windll.kernel32
        else:
            self._libc = ctypes.CDLL(None)

    def allocate_secure_tensor(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Allocate a tensor whose underlying storage resides in locked memory.
        For CPU tensors, we control the backing buffer directly. For CUDA
        tensors, we allocate on CPU first, lock, then transfer.
        """
        # Calculate byte size
        numel = 1
        for s in shape:
            numel *= s
        element_size = torch.tensor([], dtype=dtype).element_size()
        byte_size = numel * element_size

        # Allocate via ctypes (controlled allocation)
        RawArray = ctypes.c_char * byte_size
        raw = RawArray()
        address = ctypes.addressof(raw)

        # Lock the memory
        if not self._lock(address, byte_size):
            logger.warning("Could not lock tensor memory; may be paged to disk.")

        # Track allocation
        self._allocations[address] = (raw, byte_size)

        # Create a PyTorch tensor from this buffer (zero-copy for CPU)
        # We use torch.frombuffer to share memory with our locked allocation
        buffer_bytes = (ctypes.c_char * byte_size).from_address(address)
        cpu_tensor = torch.frombuffer(buffer_bytes, dtype=dtype).reshape(shape)

        # Map tensor to allocation
        self._tensor_map[id(cpu_tensor)] = address

        if device != "cpu" and torch.cuda.is_available():
            # Transfer to GPU; the CPU locked copy acts as our secure staging area
            gpu_tensor = cpu_tensor.to(device)
            # Wipe the CPU staging copy immediately
            ctypes.memset(address, 0, byte_size)
            # For GPU tensors we cannot lock VRAM, but VRAM is not pageable
            # by design. Track for cleanup on release.
            self._tensor_map[id(gpu_tensor)] = address
            return gpu_tensor

        return cpu_tensor

    def release_tensor(self, tensor: torch.Tensor):
        """Securely release a tensor: zero the storage, wipe, and unlock."""
        tensor_id = id(tensor)

        # Zero the PyTorch view first
        if tensor.is_cuda:
            tensor.zero_()
            torch.cuda.synchronize()
        else:
            tensor.zero_()

        # Wipe the backing ctypes allocation if we have one
        if tensor_id in self._tensor_map:
            address = self._tensor_map.pop(tensor_id)
            if address in self._allocations:
                raw, byte_size = self._allocations.pop(address)
                self._secure_wipe(address, byte_size)
                self._unlock(address, byte_size)

    def release_all(self):
        """Wipe and release all tracked allocations."""
        for address, (raw, byte_size) in list(self._allocations.items()):
            self._secure_wipe(address, byte_size)
            self._unlock(address, byte_size)
        self._allocations.clear()
        self._tensor_map.clear()
        logger.info("All enclave tensor allocations released and wiped.")

    # ── Internal helpers ───────────────────────────────────────────────

    def _lock(self, address: int, size: int) -> bool:
        if os.name == 'nt':
            return self._kernel32.VirtualLock(
                ctypes.c_void_p(address), ctypes.c_size_t(size)
            ) != 0
        else:
            return self._libc.mlock(
                ctypes.c_void_p(address), ctypes.c_size_t(size)
            ) == 0

    def _unlock(self, address: int, size: int):
        if os.name == 'nt':
            self._kernel32.VirtualUnlock(
                ctypes.c_void_p(address), ctypes.c_size_t(size)
            )
        else:
            self._libc.munlock(
                ctypes.c_void_p(address), ctypes.c_size_t(size)
            )

    def _secure_wipe(self, address: int, size: int):
        """3-pass secure wipe: zeros → random → zeros."""
        ptr = ctypes.c_void_p(address)
        sz = ctypes.c_size_t(size)

        # Pass 1: zeros
        ctypes.memset(address, 0, size)

        # Pass 2: random
        random_fill = os.urandom(size)
        ctypes.memmove(address, random_fill, size)
        del random_fill

        # Pass 3: zeros
        ctypes.memset(address, 0, size)
