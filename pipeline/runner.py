"""
Secure Pipeline Runner — ML-Optimized
========================================
Integrated pipeline with temporal smoothing, confidence calibration,
and all zero-trust subsystems active.
"""

import torch
import time
import logging
from typing import Iterator, Optional

from model.detector import SecureYOLO
from model.robustness import AdversarialRobustness, TemporalConsistencyChecker
from model.temporal import TemporalSmoother
from security.runtime_integrity import RuntimeIntegrityMonitor, LayerHashRegistry
from security.side_channel import TimingNoiseInjector, CUDAScheduleRandomizer, LatencyBenchmark
from security.secure_log import SecureLogger

logger = logging.getLogger("SecurePipeline")


class SecurePipelineRunner:
    """
    Executes the neural engine with all zero-trust subsystems
    and ML optimizations active.
    """

    def __init__(
        self,
        model: SecureYOLO,
        device: torch.device,
        integrity_monitor: Optional[RuntimeIntegrityMonitor] = None,
        secure_logger: Optional[SecureLogger] = None,
        enable_timing_defense: bool = True,
        enable_cuda_noise: bool = True,
        enable_temporal_smoothing: bool = True,
        target_latency_ms: float = 15.0,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Subsystems
        self.integrity_monitor = integrity_monitor
        self.secure_logger = secure_logger
        self.temporal_checker = TemporalConsistencyChecker(
            window_size=5, iou_threshold=0.15, alert_threshold=0.5
        )

        # Temporal smoothing (ML)
        self.temporal_smoother = (
            TemporalSmoother(
                ema_alpha=0.3, box_momentum=0.7,
                num_classes=model.num_classes
            ) if enable_temporal_smoothing else None
        )

        # Side-channel defenses
        self.timing_injector = (
            TimingNoiseInjector(target_latency_ms=target_latency_ms)
            if enable_timing_defense else None
        )
        self.cuda_randomizer = (
            CUDAScheduleRandomizer(num_dummy_streams=2)
            if enable_cuda_noise and torch.cuda.is_available() else None
        )
        self.latency_benchmark = LatencyBenchmark()

        # Mixed-precision (FP16 for CUDA)
        self.use_amp = device.type == 'cuda'

    def stream_secure_frames(self, frames_iterator: Iterator[torch.Tensor]):
        """Run the fully-secured ML-optimized detection loop."""
        if self.secure_logger:
            self.secure_logger.log(SecureLogger.INFERENCE_START, {
                "device": str(self.device),
                "amp_enabled": self.use_amp,
                "temporal_smoothing": self.temporal_smoother is not None,
            })

        logger.info("Starting secure inference pipeline (all subsystems + ML optimizations).")
        fps_counter = 0
        start_time = time.time()

        try:
            for frame_tensor in frames_iterator:
                # ── 1. Transfer & normalize ──────────────────────────
                t_frame = frame_tensor.to(self.device).float() / 255.0
                if t_frame.dim() == 3:
                    t_frame = t_frame.unsqueeze(0)

                # ── 2. Adversarial preprocessing ─────────────────────
                t_frame = AdversarialRobustness.preprocess_frame(
                    t_frame,
                    allow_cpu_transforms=False,
                )

                # ── 3. CUDA dummy noise (pre-inference) ──────────────
                if self.cuda_randomizer:
                    self.cuda_randomizer.pre_inference_noise()

                # ── 4. Secure inference (mixed precision) ────────────
                infer_start = time.perf_counter()

                def _infer():
                    with torch.no_grad():
                        if self.use_amp:
                            with torch.amp.autocast('cuda'):
                                return self.model.predict_padded(t_frame)
                        else:
                            return self.model.predict_padded(t_frame)

                if self.timing_injector:
                    raw_predictions = self.timing_injector.wrap_inference(_infer)
                else:
                    raw_predictions = _infer()

                infer_end = time.perf_counter()
                self.latency_benchmark.record(infer_end - infer_start)

                # ── 5. CUDA dummy noise (post-inference) ─────────────
                if self.cuda_randomizer:
                    self.cuda_randomizer.post_inference_noise()
                    self.cuda_randomizer.synchronize()

                # ── 6. Decoded/NMS detections are already confidence-filtered ──
                filtered = raw_predictions

                # ── 7. Temporal smoothing (ML) ───────────────────────
                if self.temporal_smoother:
                    filtered = self.temporal_smoother.smooth(filtered)

                # ── 8. Temporal consistency check (security) ─────────
                flat = filtered.view(filtered.shape[0], -1, 5 + self.model.num_classes)
                is_consistent, score = self.temporal_checker.check(flat[0])

                if not is_consistent and self.secure_logger:
                    self.secure_logger.log(SecureLogger.TEMPORAL_ANOMALY, {
                        "frame": fps_counter,
                        "consistency_score": round(score, 4),
                    })

                # ── 9. Runtime integrity check ───────────────────────
                if self.integrity_monitor:
                    self.integrity_monitor.step()

                # ── 10. Wipe buffers ─────────────────────────────────
                t_frame.zero_()
                del t_frame
                raw_predictions.zero_()
                del raw_predictions

                fps_counter += 1

                # ── 11. Performance logging ──────────────────────────
                if fps_counter % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed
                    timing = self.latency_benchmark.analyze()
                    logger.info(
                        f"FPS: {fps:.1f} | "
                        f"Infer: {timing['mean_ms']:.2f}ms ± {timing['std_ms']:.2f}ms | "
                        f"CV: {timing['cv_percent']:.1f}% | "
                        f"Temporal alerts: {self.temporal_checker.alert_count}"
                    )
                    start_time = time.time()
                    self.latency_benchmark.reset()

                yield filtered.cpu()

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            if self.secure_logger:
                self.secure_logger.log(SecureLogger.TAMPER_DETECTED, {
                    "error": str(e), "frame": fps_counter,
                })
            raise
        finally:
            # Secure cleanup of temporal state
            if self.temporal_smoother:
                self.temporal_smoother.reset()

            if self.secure_logger:
                self.secure_logger.log(SecureLogger.INFERENCE_END, {
                    "total_frames": fps_counter,
                    "temporal_alerts": self.temporal_checker.alert_count,
                })
            logger.info("Pipeline closed. All intermediate buffers and temporal state destroyed.")
