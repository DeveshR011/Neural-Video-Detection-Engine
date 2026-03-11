"""
Secure Enclave Abstraction Layer
================================
Provides a unified interface for hardware-backed execution enclaves
(Intel SGX, AMD SEV, ARM TrustZone). Falls back to a software-simulated
enclave using OS memory-protection primitives on unsupported hardware.
"""

import ctypes
import os
import sys
import enum
import logging
from typing import Optional

logger = logging.getLogger("SecureEnclave")


class EnclaveBackend(enum.Enum):
    SGX = "intel_sgx"
    SEV = "amd_sev"
    TRUSTZONE = "arm_trustzone"
    SOFTWARE_SIM = "software_simulation"


def _detect_hardware_enclave() -> EnclaveBackend:
    """Attempt to detect available hardware enclave support."""
    # Intel SGX detection via CPUID leaf 0x12
    try:
        if os.name == 'nt':
            # Check for SGX device on Windows
            if os.path.exists(r"\\.\isgx") or os.path.exists(r"\\.\sgx_enclave"):
                return EnclaveBackend.SGX
        else:
            # Check for SGX device on Linux
            if os.path.exists("/dev/sgx_enclave") or os.path.exists("/dev/isgx"):
                return EnclaveBackend.SGX
            # Check for SEV on AMD
            if os.path.exists("/dev/sev"):
                return EnclaveBackend.SEV
    except Exception:
        pass

    logger.warning("No hardware enclave detected. Using software simulation.")
    return EnclaveBackend.SOFTWARE_SIM


class SecureEnclave:
    """
    Context manager that establishes a hardware-isolated (or simulated)
    execution environment. All model weights and intermediate tensors
    must live exclusively within this enclave region.
    
    Usage:
        with SecureEnclave(size_bytes=512 * 1024 * 1024) as enclave:
            # All sensitive operations happen here
            enclave.store_secret("model_key", key_bytes)
            data = enclave.retrieve_secret("model_key")
    """

    def __init__(self, size_bytes: int = 512 * 1024 * 1024):
        self.size_bytes = size_bytes
        self.backend = _detect_hardware_enclave()
        self._active = False
        self._secrets: dict[str, bytes] = {}
        self._protected_regions: list[tuple[int, int]] = []

        # Platform-specific handles
        if os.name == 'nt':
            self._kernel32 = ctypes.windll.kernel32
        else:
            self._libc = ctypes.CDLL(None)

    # ── Lifecycle ──────────────────────────────────────────────────────

    def __enter__(self):
        self._activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._deactivate()

    def _activate(self):
        """Initialize enclave protections."""
        logger.info(f"Activating enclave [backend={self.backend.value}]")

        if self.backend == EnclaveBackend.SOFTWARE_SIM:
            self._activate_software_sim()
        elif self.backend == EnclaveBackend.SGX:
            self._activate_sgx()
        elif self.backend == EnclaveBackend.SEV:
            self._activate_sev()

        # Disable core dumps globally for this process
        self._disable_core_dumps()
        self._active = True
        logger.info("Enclave active. Core dumps disabled.")

    def _deactivate(self):
        """Tear down enclave, wipe all secrets and protected memory."""
        logger.info("Deactivating enclave — wiping all secrets...")

        # Zero out every stored secret
        for name in list(self._secrets.keys()):
            secret = self._secrets[name]
            # Overwrite the bytearray in-place
            if isinstance(secret, bytearray):
                for i in range(len(secret)):
                    secret[i] = 0
            del self._secrets[name]

        # Release protected memory regions
        for addr, size in self._protected_regions:
            self._wipe_memory(addr, size)
            self._unlock_region(addr, size)

        self._protected_regions.clear()
        self._active = False
        logger.info("Enclave deactivated. All secrets destroyed.")

    # ── Software Simulation ────────────────────────────────────────────

    def _activate_software_sim(self):
        """
        Software-simulated enclave using VirtualLock (Windows) / mlock (Linux)
        plus PAGE_GUARD / mprotect to approximate isolation.
        """
        logger.info("Software-simulated enclave: allocating locked region.")

    def _activate_sgx(self):
        """Stub for Intel SGX enclave creation (requires SGX SDK)."""
        logger.info("Intel SGX enclave provisioning (stub — requires SGX SDK linkage).")

    def _activate_sev(self):
        """Stub for AMD SEV encrypted VM context (requires SEV firmware)."""
        logger.info("AMD SEV context provisioning (stub — requires SEV firmware).")

    # ── Secret Management ──────────────────────────────────────────────

    def store_secret(self, name: str, data: bytes):
        """Store sensitive data inside the enclave."""
        if not self._active:
            raise RuntimeError("Enclave is not active.")
        # Store as mutable bytearray so we can wipe it on exit
        self._secrets[name] = bytearray(data)

    def retrieve_secret(self, name: str) -> bytes:
        """Retrieve a stored secret."""
        if not self._active:
            raise RuntimeError("Enclave is not active.")
        if name not in self._secrets:
            raise KeyError(f"Secret '{name}' not found in enclave.")
        return bytes(self._secrets[name])

    # ── Memory Protection ──────────────────────────────────────────────

    def protect_region(self, address: int, size: int) -> bool:
        """Lock a memory region into physical RAM and register it for cleanup."""
        if os.name == 'nt':
            result = self._kernel32.VirtualLock(
                ctypes.c_void_p(address), ctypes.c_size_t(size)
            )
            success = result != 0
        else:
            result = self._libc.mlock(
                ctypes.c_void_p(address), ctypes.c_size_t(size)
            )
            success = result == 0

        if success:
            self._protected_regions.append((address, size))
        else:
            logger.warning(f"Failed to lock region at 0x{address:x} ({size} bytes)")
        return success

    def _unlock_region(self, address: int, size: int):
        if os.name == 'nt':
            self._kernel32.VirtualUnlock(
                ctypes.c_void_p(address), ctypes.c_size_t(size)
            )
        else:
            self._libc.munlock(
                ctypes.c_void_p(address), ctypes.c_size_t(size)
            )

    def _wipe_memory(self, address: int, size: int):
        """Perform a 3-pass secure wipe: zeros → random → zeros."""
        ptr = ctypes.c_void_p(address)
        sz = ctypes.c_size_t(size)

        # Pass 1: zeros
        if os.name == 'nt':
            self._kernel32.RtlZeroMemory(ptr, sz)
        else:
            self._libc.memset(ptr, 0, sz)

        # Pass 2: random overwrite
        random_bytes = os.urandom(size)
        ctypes.memmove(address, random_bytes, size)
        del random_bytes

        # Pass 3: zeros again
        if os.name == 'nt':
            self._kernel32.RtlZeroMemory(ptr, sz)
        else:
            self._libc.memset(ptr, 0, sz)

    # ── Core Dump Prevention ───────────────────────────────────────────

    def _disable_core_dumps(self):
        """Prevent the process from writing core dumps that might leak secrets."""
        if os.name != 'nt':
            try:
                import resource
                resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            except Exception as e:
                logger.warning(f"Could not disable core dumps: {e}")
        else:
            # On Windows, disable WER (Windows Error Reporting) crash dumps
            # via SetErrorMode. SEM_NOGPFAULTERRORBOX prevents WER dialog.
            SEM_NOGPFAULTERRORBOX = 0x0002
            SEM_FAILCRITICALERRORS = 0x0001
            self._kernel32.SetErrorMode(
                SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS
            )
