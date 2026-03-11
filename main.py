"""
Neural Video Detection Engine — Main Orchestrator (ML-Optimized)
==================================================================
Full boot-to-shutdown secure lifecycle with ML enhancements:
1. Device key derivation (TPM / fingerprint)
2. Enclave provisioning
3. Tamper-evident logging
4. Secure model load + integrity registration
5. Watermark verification
6. Architecture morphing
7. Runtime integrity registration
8. Secure pipeline with temporal smoothing
9. Extended ML benchmark suite
10. Secure shutdown
"""

import os
import torch
import logging
from typing import Iterator

# Crypto
from crypto.device_keys import TPMKeyManager
from crypto.model_cipher import ModelCipher

# Model
from model.detector import SecureYOLO
from model.watermark import WatermarkEmbedder, WatermarkVerifier

# Security
from security.enclave import SecureEnclave
from security.architecture_morph import ArchitectureMorpher
from security.runtime_integrity import LayerHashRegistry, RuntimeIntegrityMonitor
from security.secure_log import SecureLogger

# Pipeline
from pipeline.runner import SecurePipelineRunner

# Benchmarks
from benchmarks.benchmark import BenchmarkSuite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("MainControl")


def mock_video_stream(num_frames: int = 100) -> Iterator[torch.Tensor]:
    """Simulate a decrypted video stream in-memory."""
    for _ in range(num_frames):
        yield torch.randint(0, 256, (3, 640, 640), dtype=torch.uint8)


def main():
    logger.info("=" * 72)
    logger.info("  ENCRYPTED NEURAL VIDEO DETECTION ENGINE")
    logger.info("  Zero-Trust Architecture v3.0 — ML-Optimized")
    logger.info("=" * 72)

    # ══ Phase 1: Device Key Derivation ════════════════════════════════
    logger.info("[Phase 1] Deriving device-bound cryptographic keys...")
    key_manager = TPMKeyManager()
    key_manager.initialize()

    model_key = key_manager.derive_key("model_encryption")
    log_key = key_manager.derive_key("log_integrity")
    watermark_key = key_manager.derive_key("watermark_signature")
    device_fp = key_manager.fingerprint
    logger.info(f"  Device: {device_fp[:8].hex()}... | Keys: model, log, watermark")

    # ══ Phase 2: Secure Enclave ═══════════════════════════════════════
    logger.info("[Phase 2] Provisioning secure enclave...")

    with SecureEnclave(size_bytes=512 * 1024 * 1024) as enclave:
        enclave.store_secret("model_key", model_key)
        enclave.store_secret("watermark_key", watermark_key)

        # ══ Phase 3: Secure Logging ═══════════════════════════════════
        logger.info("[Phase 3] Initializing secure logging...")
        os.makedirs("secure_enclave", exist_ok=True)
        secure_logger = SecureLogger(
            hmac_key=log_key,
            log_path="secure_enclave/audit.log",
        )
        secure_logger.log(SecureLogger.SYSTEM_BOOT, {
            "device_fp": device_fp[:8].hex(),
            "enclave_backend": enclave.backend.value,
            "architecture": "CSP+BiFPN v3.0",
        })

        # ══ Phase 4: Model Load ═══════════════════════════════════════
        logger.info("[Phase 4] Loading ML-optimized model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"  Compute device: {device}")

        # Create model with new CSP+BiFPN architecture
        temp_model = SecureYOLO(num_classes=80, fpn_channels=128)
        param_count = sum(p.numel() for p in temp_model.parameters())
        logger.info(f"  Architecture: CSP+BiFPN | Params: {param_count:,}")

        # Encrypt → Decrypt roundtrip
        torch.save(temp_model.state_dict(), "secure_enclave/raw_model.pt")
        mc = ModelCipher(model_key, device_fingerprint=device_fp)
        mc.encrypt_model(
            "secure_enclave/raw_model.pt",
            "secure_enclave/detector.enc",
            "secure_enclave/detector.hash",
        )
        os.remove("secure_enclave/raw_model.pt")

        state_dict = mc.load_secure_model(
            "secure_enclave/detector.enc",
            "secure_enclave/detector.hash",
        )

        model = SecureYOLO(num_classes=80, fpn_channels=128)
        model.load_state_dict(state_dict)

        for k in state_dict:
            state_dict[k].zero_()
        del state_dict

        secure_logger.log(SecureLogger.MODEL_LOAD, {
            "integrity": "verified",
            "device_bound": True,
            "architecture": "CSP+BiFPN",
            "parameters": param_count,
        })

        # ══ Phase 5: Watermark ════════════════════════════════════════
        logger.info("[Phase 5] Embedding neural watermark...")
        embedder = WatermarkEmbedder(watermark_key)
        model = embedder.embed(model, strength=0.001)

        verifier = WatermarkVerifier(watermark_key)
        wm_valid, wm_accuracy = verifier.verify(model)

        secure_logger.log(SecureLogger.WATERMARK_VERIFIED, {
            "valid": wm_valid,
            "accuracy": round(wm_accuracy, 4),
        })

        # ══ Phase 6: Architecture Morphing ════════════════════════════
        logger.info("[Phase 6] Applying per-session architecture morphing...")
        morpher = ArchitectureMorpher()
        model = morpher.morph(model)

        # ══ Phase 7: Runtime Integrity ════════════════════════════════
        logger.info("[Phase 7] Registering runtime integrity hashes...")
        registry = LayerHashRegistry()
        registry.register_model(model)

        integrity_monitor = RuntimeIntegrityMonitor(
            model=model,
            registry=registry,
            check_probability=0.05,
            check_interval=10,
        )

        # ══ Phase 8: Secure Pipeline ══════════════════════════════════
        logger.info("[Phase 8] Starting ML-optimized secure inference pipeline...")
        secure_logger.log(SecureLogger.ENCLAVE_ENTER)

        pipeline = SecurePipelineRunner(
            model=model,
            device=device,
            integrity_monitor=integrity_monitor,
            secure_logger=secure_logger,
            enable_timing_defense=True,
            enable_cuda_noise=torch.cuda.is_available(),
            enable_temporal_smoothing=True,
            target_latency_ms=15.0,
        )

        frame_generator = mock_video_stream(num_frames=90)

        try:
            for i, result in enumerate(pipeline.stream_secure_frames(frame_generator)):
                result.zero_()
                del result
        except KeyboardInterrupt:
            logger.warning("Pipeline interrupted by user.")

        # ══ Phase 9: Extended ML Benchmarks ═══════════════════════════
        logger.info("[Phase 9] Running extended ML benchmark suite...")
        bench = BenchmarkSuite(model, device)
        bench.run_all(num_frames=50, output_path="secure_enclave/ml_benchmark_results.json")
        bench.print_summary()

        # ══ Phase 10: Secure Shutdown ═════════════════════════════════
        logger.info("[Phase 10] Initiating secure shutdown...")
        chain_valid, bad_seq = secure_logger.verify_chain()
        secure_logger.log(SecureLogger.SYSTEM_SHUTDOWN, {
            "log_chain_valid": chain_valid,
            "total_entries": secure_logger.entry_count,
            "integrity_stats": integrity_monitor.get_stats(),
        })

        logger.info(f"  Log chain: {'✅ VALID' if chain_valid else '❌ BROKEN'}")
        logger.info(f"  Total log entries: {secure_logger.entry_count}")

    logger.info("=" * 72)
    logger.info("  SHUTDOWN COMPLETE — All secrets destroyed")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
