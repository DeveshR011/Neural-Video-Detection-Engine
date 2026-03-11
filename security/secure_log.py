"""
Tamper-Evident Secure Logging
==============================
Cryptographic hash-chain logger. Each log entry contains a sequence
number, timestamp, event type, data payload, prior hash, and current
hash — forming an immutable forensic chain. Keyed via HMAC-SHA256
bound to the device-derived log key.
"""

import hashlib
import hmac
import json
import time
import os
import logging
from typing import Optional

logger = logging.getLogger("SecureLog")


class SecureLogEntry:
    """Represents a single tamper-evident log entry."""

    def __init__(self, seq: int, timestamp: float, event: str,
                 data: dict, prev_hash: str, current_hash: str):
        self.seq = seq
        self.timestamp = timestamp
        self.event = event
        self.data = data
        self.prev_hash = prev_hash
        self.current_hash = current_hash

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "timestamp": self.timestamp,
            "event": self.event,
            "data": self.data,
            "prev_hash": self.prev_hash,
            "current_hash": self.current_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict) -> "SecureLogEntry":
        return cls(
            seq=d["seq"],
            timestamp=d["timestamp"],
            event=d["event"],
            data=d["data"],
            prev_hash=d["prev_hash"],
            current_hash=d["current_hash"],
        )


class SecureLogger:
    """
    Hash-chain logger that guarantees forensic integrity.
    
    Each entry's hash = HMAC-SHA256(key, seq | timestamp | event | data | prev_hash)
    
    Events:
        SYSTEM_BOOT, MODEL_LOAD, INFERENCE_START, INFERENCE_END,
        INTEGRITY_CHECK_PASS, INTEGRITY_CHECK_FAIL, TAMPER_DETECTED,
        TEMPORAL_ANOMALY, WATERMARK_VERIFIED, SYSTEM_SHUTDOWN
    """

    # Standard event types
    SYSTEM_BOOT = "SYSTEM_BOOT"
    MODEL_LOAD = "MODEL_LOAD"
    INFERENCE_START = "INFERENCE_START"
    INFERENCE_END = "INFERENCE_END"
    INTEGRITY_PASS = "INTEGRITY_CHECK_PASS"
    INTEGRITY_FAIL = "INTEGRITY_CHECK_FAIL"
    TAMPER_DETECTED = "TAMPER_DETECTED"
    TEMPORAL_ANOMALY = "TEMPORAL_ANOMALY"
    WATERMARK_VERIFIED = "WATERMARK_VERIFIED"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    ENCLAVE_ENTER = "ENCLAVE_ENTER"
    ENCLAVE_EXIT = "ENCLAVE_EXIT"

    def __init__(self, hmac_key: bytes, log_path: Optional[str] = None):
        """
        Args:
            hmac_key: Device-derived key for HMAC computation
            log_path: Path to write the log file (optional; if None, in-memory only)
        """
        self.hmac_key = hmac_key
        self.log_path = log_path
        self._chain: list[SecureLogEntry] = []
        self._seq_counter = 0
        self._prev_hash = "0" * 64  # Genesis hash

        # Initialize log file
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, 'w') as f:
                f.write("")  # Truncate

    def log(self, event: str, data: Optional[dict] = None):
        """
        Append a new entry to the hash chain.
        
        Args:
            event: Event type string
            data: Arbitrary data payload
        """
        if data is None:
            data = {}

        self._seq_counter += 1
        timestamp = time.time()

        # Compute HMAC-SHA256 hash
        content = f"{self._seq_counter}|{timestamp}|{event}|{json.dumps(data, sort_keys=True)}|{self._prev_hash}"
        current_hash = hmac.new(
            self.hmac_key, content.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        entry = SecureLogEntry(
            seq=self._seq_counter,
            timestamp=timestamp,
            event=event,
            data=data,
            prev_hash=self._prev_hash,
            current_hash=current_hash,
        )

        self._chain.append(entry)
        self._prev_hash = current_hash

        # Persist to disk
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(entry.to_json() + "\n")

        logger.debug(f"SecureLog [{self._seq_counter}] {event}: {current_hash[:16]}...")

    def verify_chain(self) -> tuple[bool, Optional[int]]:
        """
        Verify the integrity of the entire log chain.
        
        Returns:
            (is_valid, first_invalid_seq) — if invalid, returns the
            sequence number of the first tampered entry.
        """
        prev_hash = "0" * 64  # Genesis

        entries = self._chain
        if self.log_path and not entries:
            entries = self._load_from_file()

        for entry in entries:
            # Recompute expected hash
            content = (
                f"{entry.seq}|{entry.timestamp}|{entry.event}|"
                f"{json.dumps(entry.data, sort_keys=True)}|{prev_hash}"
            )
            expected_hash = hmac.new(
                self.hmac_key, content.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            # Constant-time comparison
            if not hmac.compare_digest(expected_hash, entry.current_hash):
                logger.critical(
                    f"LOG CHAIN BROKEN at seq={entry.seq}! "
                    f"Expected {expected_hash[:16]}..., "
                    f"got {entry.current_hash[:16]}..."
                )
                return False, entry.seq

            if entry.prev_hash != prev_hash:
                logger.critical(
                    f"LOG CHAIN prev_hash MISMATCH at seq={entry.seq}!"
                )
                return False, entry.seq

            prev_hash = entry.current_hash

        logger.info(f"Log chain verified: {len(entries)} entries, all valid.")
        return True, None

    def _load_from_file(self) -> list[SecureLogEntry]:
        """Load entries from the log file."""
        entries = []
        if self.log_path and os.path.exists(self.log_path):
            with open(self.log_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(SecureLogEntry.from_dict(json.loads(line)))
        return entries

    @property
    def entry_count(self) -> int:
        return self._seq_counter

    def get_chain_summary(self) -> dict:
        return {
            "total_entries": self._seq_counter,
            "latest_hash": self._prev_hash,
            "log_path": self.log_path,
        }
