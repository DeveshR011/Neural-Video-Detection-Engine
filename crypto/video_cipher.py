"""
AES-256-GCM Video Cipher — Device-Bound
=========================================
Encrypts/decrypts video files with device fingerprint bound into the
AES-GCM Associated Authenticated Data (AAD).
"""

import os
import logging

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from crypto.memory import SecureBuffer

logger = logging.getLogger("VideoCipher")


class VideoCipher:
    """Handles AES-256-GCM encryption and decryption of video streams."""

    def __init__(self, device_fingerprint: bytes = b""):
        self._keys: dict[str, AESGCM] = {}
        self.device_fingerprint = device_fingerprint

    def add_key(self, video_id: str, key: bytes):
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        self._keys[video_id] = AESGCM(key)

    def encrypt_file(self, video_id: str, input_path: str, output_path: str):
        if video_id not in self._keys:
            raise KeyError(f"Key for video {video_id} not found.")

        aes_gcm = self._keys[video_id]
        nonce = os.urandom(12)

        with open(input_path, 'rb') as f:
            data = f.read()

        aad = self.device_fingerprint if self.device_fingerprint else None
        ciphertext = aes_gcm.encrypt(nonce, data, aad)

        with open(output_path, 'wb') as f:
            f.write(nonce)
            f.write(ciphertext)

        logger.info(f"Video '{video_id}' encrypted → {output_path}")

    def decrypt_to_secure_buffer(self, video_id: str, encrypted_path: str) -> SecureBuffer:
        """Decrypt video data into a locked memory buffer."""
        if video_id not in self._keys:
            raise KeyError(f"Key for video {video_id} not found.")

        aes_gcm = self._keys[video_id]

        with open(encrypted_path, 'rb') as f:
            nonce = f.read(12)
            ciphertext = f.read()

        aad = self.device_fingerprint if self.device_fingerprint else None
        plaintext = aes_gcm.decrypt(nonce, ciphertext, aad)

        buffer = SecureBuffer(len(plaintext))
        buffer.write(plaintext)
        del plaintext

        return buffer
