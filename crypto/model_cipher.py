"""
AES-256-GCM Model Cipher — Device-Bound
=========================================
Encrypts/decrypts model weights with device fingerprint bound into
the AES-GCM Associated Authenticated Data (AAD). Cross-device
decryption will fail with an authentication error.
"""

import os
import hashlib
import io
import logging

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import torch

from crypto.memory import SecureBuffer

logger = logging.getLogger("ModelCipher")


class ModelCipher:
    """Handles secure loading, integrity verification, and decryption of model weights."""

    def __init__(self, key: bytes, device_fingerprint: bytes = b""):
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        self.aes_gcm = AESGCM(key)
        self.device_fingerprint = device_fingerprint

    def encrypt_model(self, model_path: str, output_path: str, hash_path: str):
        """Encrypt a PyTorch model file and generate a SHA-256 hash."""
        with open(model_path, 'rb') as f:
            data = f.read()

        self._encrypt_bytes(data, output_path, hash_path)

    def encrypt_state_dict(self, state_dict: dict, output_path: str, hash_path: str):
        """Encrypt a state_dict directly from memory without writing plaintext to disk."""
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        data = buffer.getvalue()
        self._encrypt_bytes(data, output_path, hash_path)

    def _encrypt_bytes(self, data: bytes, output_path: str, hash_path: str):
        """Common byte-oriented encryption path for model artifacts."""

        # Integrity hash (pre-encryption)
        model_hash = hashlib.sha256(data).digest()
        with open(hash_path, 'wb') as f:
            f.write(model_hash)

        # Encrypt with device fingerprint as AAD
        nonce = os.urandom(12)
        aad = self.device_fingerprint if self.device_fingerprint else None
        ciphertext = self.aes_gcm.encrypt(nonce, data, aad)

        with open(output_path, 'wb') as f:
            f.write(nonce)
            f.write(ciphertext)

        logger.info(f"Model encrypted → {output_path} (device-bound AAD: {len(self.device_fingerprint)} bytes)")

    def load_secure_model(self, encrypted_path: str, hash_path: str) -> dict:
        """
        Decrypt model into memory, verify integrity, and load state_dict.
        Will fail if decrypted on a different device (AAD mismatch).
        """
        with open(encrypted_path, 'rb') as f:
            nonce = f.read(12)
            ciphertext = f.read()

        # Decrypt with device-bound AAD
        aad = self.device_fingerprint if self.device_fingerprint else None
        try:
            plaintext = self.aes_gcm.decrypt(nonce, ciphertext, aad)
        except Exception:
            raise ValueError(
                "Decryption failed. Invalid key, tampered model, or wrong device."
            )

        # Verify integrity
        computed_hash = hashlib.sha256(plaintext).digest()
        with open(hash_path, 'rb') as f:
            expected_hash = f.read()

        if computed_hash != expected_hash:
            del plaintext
            raise RuntimeError(
                "CRITICAL: Model integrity verification failed. "
                "Potential tampering detected. Aborting."
            )

        # Load state dictionary from in-memory buffer
        state_dict = torch.load(io.BytesIO(plaintext), weights_only=True)

        del plaintext
        logger.info("Model decrypted and integrity-verified successfully.")
        return state_dict
