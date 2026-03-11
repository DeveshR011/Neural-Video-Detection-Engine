"""
Device-Bound Cryptographic Key Architecture
=============================================
Replaces static encryption keys with TPM-sealed, HKDF-derived keys
bound to a unique device fingerprint. Prevents cross-device decryption
even if encrypted files are exfiltrated.
"""

import hashlib
import hmac
import os
import sys
import json
import logging
import platform
import subprocess
from typing import Optional

from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

logger = logging.getLogger("DeviceKeys")


class DeviceFingerprint:
    """
    Collects hardware identifiers to build a unique, deterministic
    device identity hash (SHA-256). On Windows uses WMI; on Linux
    reads /sys and DMI tables.
    """

    @staticmethod
    def collect() -> bytes:
        """Return a 32-byte SHA-256 device fingerprint."""
        components: list[str] = []

        try:
            if os.name == 'nt':
                components.extend(DeviceFingerprint._windows_ids())
            else:
                components.extend(DeviceFingerprint._linux_ids())
        except Exception as e:
            logger.warning(f"Partial fingerprint collection: {e}")

        # Always include platform basics as a fallback
        components.append(platform.node())
        components.append(platform.machine())
        components.append(platform.processor())

        combined = "|".join(components)
        fingerprint = hashlib.sha256(combined.encode("utf-8")).digest()
        logger.info(f"Device fingerprint computed: {fingerprint[:8].hex()}...")
        return fingerprint

    @staticmethod
    def _windows_ids() -> list[str]:
        """Collect identifiers via WMI on Windows."""
        ids: list[str] = []
        try:
            import wmi
            c = wmi.WMI()

            # CPU serial
            for cpu in c.Win32_Processor():
                ids.append(f"cpu:{cpu.ProcessorId}")

            # Motherboard serial
            for board in c.Win32_BaseBoard():
                ids.append(f"mb:{board.SerialNumber}")

            # Disk serial
            for disk in c.Win32_DiskDrive():
                ids.append(f"disk:{disk.SerialNumber}")

        except ImportError:
            logger.warning("wmi package not installed; using fallback fingerprint")
            # Fallback: use Windows machine GUID from registry
            try:
                result = subprocess.run(
                    ["reg", "query",
                     r"HKLM\SOFTWARE\Microsoft\Cryptography",
                     "/v", "MachineGuid"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        if "MachineGuid" in line:
                            ids.append(f"guid:{line.split()[-1]}")
            except Exception:
                pass
        return ids

    @staticmethod
    def _linux_ids() -> list[str]:
        """Collect identifiers from /sys and DMI on Linux."""
        ids: list[str] = []
        paths = [
            ("/sys/class/dmi/id/product_uuid", "uuid"),
            ("/sys/class/dmi/id/board_serial", "mb"),
            ("/etc/machine-id", "machine"),
        ]
        for path, prefix in paths:
            try:
                with open(path, 'r') as f:
                    ids.append(f"{prefix}:{f.read().strip()}")
            except Exception:
                pass
        return ids


class TPMKeyManager:
    """
    Manages a master secret sealed to the TPM. If TPM is unavailable,
    derives a master secret from the device fingerprint combined with
    an optional user-provided passphrase.
    """

    def __init__(self, passphrase: Optional[str] = None):
        self.fingerprint = DeviceFingerprint.collect()
        self._master_secret: Optional[bytes] = None
        self._passphrase = passphrase

    def initialize(self) -> bytes:
        """Obtain the master secret, either from TPM or derived locally."""
        tpm_secret = self._try_tpm_unseal()
        if tpm_secret is not None:
            self._master_secret = tpm_secret
            logger.info("Master secret retrieved from TPM.")
        else:
            self._master_secret = self._derive_local_master()
            logger.info("Master secret derived from device fingerprint (no TPM).")
        return self._master_secret

    def derive_key(self, purpose: str, context: bytes = b"", length: int = 32) -> bytes:
        """
        Derive a purpose-specific key via HKDF-SHA256.
        
        Args:
            purpose: Human-readable key purpose (e.g., "model_encryption")
            context: Additional context bytes bound into the derivation
            length: Output key length in bytes
        """
        if self._master_secret is None:
            raise RuntimeError("TPMKeyManager not initialized. Call initialize() first.")

        info = purpose.encode("utf-8") + b"|" + context
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=length,
            salt=self.fingerprint,
            info=info,
        )
        derived = hkdf.derive(self._master_secret)
        return derived

    def _try_tpm_unseal(self) -> Optional[bytes]:
        """
        Attempt to unseal the master secret from a TPM. On Windows,
        this would use the TBS (TPM Base Services) API. On Linux,
        this would use tpm2-tools.
        """
        try:
            if os.name == 'nt':
                return self._tpm_windows()
            else:
                return self._tpm_linux()
        except Exception as e:
            logger.debug(f"TPM not available: {e}")
            return None

    def _tpm_windows(self) -> Optional[bytes]:
        """Stub for Windows TBS API interaction."""
        # In a production system, use ctypes to call TBS APIs:
        #   Tbsi_Context_Create, Tbsi_Get_OwnerAuth, etc.
        # For now, return None to indicate TPM unavailable.
        return None

    def _tpm_linux(self) -> Optional[bytes]:
        """Try to unseal via tpm2-tools."""
        sealed_path = "/var/lib/neural-engine/tpm_sealed.bin"
        if not os.path.exists(sealed_path):
            return None
        try:
            result = subprocess.run(
                ["tpm2_unseal", "-c", "0x81000001", "-o", "-"],
                capture_output=True, timeout=10
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout[:32]
        except FileNotFoundError:
            pass
        return None

    def _derive_local_master(self) -> bytes:
        """
        Derive a master secret from the device fingerprint and optional
        passphrase. This is the fallback when no TPM is available.
        """
        salt = self.fingerprint
        ikm = self.fingerprint
        if self._passphrase:
            ikm = ikm + self._passphrase.encode("utf-8")

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"neural-engine-master-v1",
        )
        return hkdf.derive(ikm)
