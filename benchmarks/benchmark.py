"""
Benchmark Suite
===============
Separates synthetic performance stress tests from real-data detection metrics.
Quality metrics are only reported when labeled COCO data is provided.
"""

import copy
import json
import logging
import os
import statistics
import time
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from model.detector import SecureYOLO
from model.robustness import AdversarialRobustness
from model.temporal import TemporalSmoother
from model.calibration import CalibrationMetrics
from security.runtime_integrity import LayerHashRegistry
from security.architecture_morph import ArchitectureMorpher

logger = logging.getLogger("Benchmark")


class BenchmarkSuite:
    """Benchmark suite with explicit quality/performance separation."""

    def __init__(self, model: SecureYOLO, device: torch.device):
        self.model = model
        self.device = device
        self.results: dict = {}

    def run_all(
        self,
        num_frames: int = 100,
        output_path: Optional[str] = None,
        eval_data_root: Optional[str] = None,
        eval_split: str = "val",
        eval_max_images: int = 100,
        include_experimental: bool = False,
    ) -> dict:
        logger.info("Starting benchmark suite")

        self.results["latency"] = self._benchmark_latency(num_frames)
        self.results["memory"] = self._benchmark_memory()
        self.results["temporal_consistency"] = self._benchmark_temporal()
        if include_experimental:
            self.results["morph_variance"] = self._benchmark_morph_variance()
        else:
            self.results["morph_variance"] = {
                "status": "skipped",
                "message": "Set include_experimental=True to run morph variance.",
            }
        self.results["extraction_resistance"] = self._benchmark_extraction()
        self.results["baseline_comparison"] = self._benchmark_three_way(num_frames)

        if eval_data_root:
            logger.info("Running real-data evaluation metrics")
            base_eval = self._evaluate_on_coco(eval_data_root, eval_split, eval_max_images)
            self.results["detection_quality"] = base_eval
            self.results["calibration"] = self._calibration_from_eval(base_eval)
            self.results["adversarial_eval"] = self._benchmark_adversarial_on_eval(
                eval_data_root, eval_split, max(20, min(eval_max_images, 60))
            )
        else:
            self.results["detection_quality"] = {
                "status": "skipped_no_eval_data",
                "message": "Pass eval_data_root to compute mAP/precision/recall on labeled data.",
            }
            self.results["calibration"] = {
                "status": "skipped_no_eval_data",
                "message": "ECE is only computed from matched detections on labeled data.",
            }
            self.results["adversarial_eval"] = {
                "status": "skipped_no_eval_data",
                "message": "Adversarial quality metrics require labeled evaluation data.",
            }

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Benchmark results saved -> {output_path}")

        return self.results

    def _benchmark_latency(self, num_frames: int) -> dict:
        logger.info("Benchmark: latency (synthetic frames, performance-only)")
        self.model.eval()
        latencies = []

        for _ in range(10):
            dummy = torch.randn(1, 3, 640, 640, device=self.device)
            with torch.no_grad():
                _ = self.model.predict(dummy, conf_threshold=0.35, max_detections=200, pre_nms_topk=800)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

        for _ in range(num_frames):
            frame = torch.randn(1, 3, 640, 640, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model.predict(frame, conf_threshold=0.35, max_detections=200, pre_nms_topk=800)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

        mean_ms = statistics.mean(latencies) * 1000
        std_ms = statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0.0
        fps = 1000 / mean_ms if mean_ms > 0 else 0

        return {
            "mode": "performance_only",
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(std_ms, 3),
            "min_ms": round(min(latencies) * 1000, 3),
            "max_ms": round(max(latencies) * 1000, 3),
            "fps": round(fps, 1),
            "meets_30fps": fps >= 30,
        }

    def _benchmark_memory(self) -> dict:
        logger.info("Benchmark: memory profile")
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())

        frame = torch.randn(1, 3, 640, 640, device=self.device)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            with torch.no_grad():
                _ = self.model.predict(frame)
            peak = torch.cuda.max_memory_allocated()
            act_bytes = peak - before
        else:
            act_bytes = 0

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable,
            "parameter_memory_mb": round(param_bytes / (1024 ** 2), 2),
            "peak_activation_memory_mb": round(act_bytes / (1024 ** 2), 2),
        }

    def _benchmark_temporal(self) -> dict:
        logger.info("Benchmark: temporal consistency (synthetic sequence)")
        self.model.eval()
        smoother = TemporalSmoother(ema_alpha=0.3, box_momentum=0.7, num_classes=self.model.num_classes)

        raw_stability = []
        smooth_stability = []
        prev_raw = None
        prev_smooth = None

        for _ in range(8):
            frame = torch.rand(1, 3, 640, 640, device=self.device)
            with torch.no_grad():
                pred = self.model.predict(frame)
            smoothed = smoother.smooth(pred)

            if prev_raw is not None:
                raw_diff = torch.abs(pred[0, :, :4] - prev_raw[0, :, :4]).mean().item()
                smooth_diff = torch.abs(smoothed[0, :, :4] - prev_smooth[0, :, :4]).mean().item()
                raw_stability.append(raw_diff)
                smooth_stability.append(smooth_diff)

            prev_raw = pred.clone()
            prev_smooth = smoothed.clone()

        smoother.reset()

        raw_mean = statistics.mean(raw_stability) if raw_stability else 0.0
        smooth_mean = statistics.mean(smooth_stability) if smooth_stability else 0.0
        improve = (1 - (smooth_mean / max(raw_mean, 1e-6))) * 100 if raw_stability else 0.0

        return {
            "mode": "synthetic_sequence_stability",
            "raw_box_delta": round(raw_mean, 6),
            "smoothed_box_delta": round(smooth_mean, 6),
            "delta_reduction_pct": round(improve, 2),
        }

    def _benchmark_morph_variance(self, n_seeds: int = 5) -> dict:
        logger.info("Benchmark: morph variance (experimental)")
        test_frame = torch.rand(1, 3, 640, 640, device=self.device)
        outputs = []

        for seed in range(n_seeds):
            model_copy = copy.deepcopy(self.model)
            morpher = ArchitectureMorpher(session_seed=seed.to_bytes(4, "big"), enabled=True)
            morpher.morph(model_copy)
            model_copy.eval()
            with torch.no_grad():
                out = model_copy.predict(test_frame)
            outputs.append(out.cpu())
            del model_copy

        sims = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                sim = nn.functional.cosine_similarity(
                    outputs[i].flatten().unsqueeze(0),
                    outputs[j].flatten().unsqueeze(0),
                ).item()
                sims.append(sim)

        return {
            "status": "experimental",
            "n_permutations": n_seeds,
            "mean_cosine_similarity": round(statistics.mean(sims), 4) if sims else 0.0,
            "std_cosine_similarity": round(statistics.stdev(sims), 4) if len(sims) > 1 else 0.0,
            "warning": "Morphing does not have a strict output-invariance guarantee for this topology.",
        }

    def _benchmark_extraction(self) -> dict:
        logger.info("Benchmark: extraction resistance inventory")
        registry = LayerHashRegistry()
        registry.register_model(self.model)

        return {
            "total_monitored_layers": len(registry.get_registered_names()),
            "random_projection_active": hasattr(self.model, "random_projection"),
            "feature_denoising_active": hasattr(self.model, "denoise_p3"),
            "bifpn_active": hasattr(self.model, "bifpn"),
            "multi_scale_heads": hasattr(self.model, "head_p3"),
            "architecture_morphing_default": "disabled",
        }

    def _benchmark_three_way(self, num_frames: int) -> dict:
        logger.info("Benchmark: three-way latency comparison (performance-only)")

        raw_lats = []
        for _ in range(num_frames):
            frame = torch.randn(1, 3, 640, 640, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model.predict(frame, conf_threshold=0.35, max_detections=200, pre_nms_topk=800)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            raw_lats.append(time.perf_counter() - start)

        secure_lats = []
        for _ in range(num_frames):
            frame = torch.randn(1, 3, 640, 640, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            frame = AdversarialRobustness.preprocess_frame(frame, allow_cpu_transforms=False)
            with torch.no_grad():
                _ = self.model.predict(frame, conf_threshold=0.35, max_detections=200, pre_nms_topk=800)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            secure_lats.append(time.perf_counter() - start)

        smoother = TemporalSmoother(num_classes=self.model.num_classes)
        ml_lats = []
        for _ in range(num_frames):
            frame = torch.randn(1, 3, 640, 640, device=self.device)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            frame = AdversarialRobustness.preprocess_frame(frame, allow_cpu_transforms=False)
            with torch.no_grad():
                pred = self.model.predict(frame, conf_threshold=0.35, max_detections=200, pre_nms_topk=800)
            _ = smoother.smooth(pred)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ml_lats.append(time.perf_counter() - start)
        smoother.reset()

        raw_ms = statistics.mean(raw_lats) * 1000
        sec_ms = statistics.mean(secure_lats) * 1000
        ml_ms = statistics.mean(ml_lats) * 1000

        return {
            "mode": "performance_only",
            "raw": {"ms": round(raw_ms, 3), "fps": round(1000 / raw_ms, 1) if raw_ms > 0 else 0},
            "secure": {"ms": round(sec_ms, 3), "fps": round(1000 / sec_ms, 1) if sec_ms > 0 else 0},
            "secure_ml": {"ms": round(ml_ms, 3), "fps": round(1000 / ml_ms, 1) if ml_ms > 0 else 0},
            "security_overhead_pct": round((sec_ms - raw_ms) / raw_ms * 100, 2) if raw_ms > 0 else 0,
            "ml_overhead_pct": round((ml_ms - sec_ms) / sec_ms * 100, 2) if sec_ms > 0 else 0,
        }

    def _load_coco_eval_records(self, data_root: str, split: str, max_images: int) -> list[dict]:
        import cv2

        ann_file = os.path.join(data_root, "annotations", f"instances_{split}2017.json")
        img_dir = os.path.join(data_root, "images", f"{split}2017")

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Missing annotation file: {ann_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Missing image directory: {img_dir}")

        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)

        cat_ids = [c["id"] for c in coco["categories"]]
        cat_to_idx = {cid: i for i, cid in enumerate(cat_ids)}

        images = {img["id"]: img for img in coco["images"]}
        ann_by_img: dict[int, list] = {}
        for ann in coco["annotations"]:
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        records = []
        for img_id, anns in ann_by_img.items():
            if len(records) >= max_images:
                break
            info = images.get(img_id)
            if info is None:
                continue

            path = os.path.join(img_dir, info["file_name"])
            img = cv2.imread(path)
            if img is None:
                continue

            h, w = img.shape[:2]
            gt = []
            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2.0) / max(w, 1)
                cy = (y + bh / 2.0) / max(h, 1)
                nw = bw / max(w, 1)
                nh = bh / max(h, 1)
                cls = cat_to_idx.get(ann["category_id"])
                if cls is None:
                    continue
                gt.append((np.array([cx, cy, nw, nh], dtype=np.float32), int(cls)))

            if gt:
                records.append({"path": path, "gt": gt})

        return records

    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        out = boxes.copy()
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out

    def _iou_np(self, a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter + 1e-7
        return inter / union

    def _compute_ap(self, recall: np.ndarray, precision: np.ndarray) -> float:
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    def _evaluate_on_coco(
        self,
        data_root: str,
        split: str,
        max_images: int,
        perturb: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> dict:
        import cv2

        self.model.eval()
        records = self._load_coco_eval_records(data_root, split, max_images)
        if not records:
            return {
                "status": "no_records",
                "message": "No labeled images were available for evaluation.",
            }

        detections_by_class: dict[int, list] = {}
        gt_count_by_class: dict[int, int] = {}
        per_image_matches = []

        for img_idx, rec in enumerate(records):
            img = cv2.imread(rec["path"])
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (640, 640))
            inp = torch.from_numpy(resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device) / 255.0

            if perturb is not None:
                inp = perturb(inp)

            with torch.no_grad():
                pred = self.model.predict(inp)[0].detach().cpu().numpy()

            conf = pred[:, 4]
            cls = np.argmax(pred[:, 5:], axis=1)
            keep = (conf > 1e-6) & (pred[:, 2] > 0) & (pred[:, 3] > 0)
            pred_boxes = pred[keep, :4]
            pred_scores = conf[keep]
            pred_cls = cls[keep]
            pred_xyxy = self._xywh_to_xyxy(pred_boxes)

            gt_boxes = np.array([g[0] for g in rec["gt"]], dtype=np.float32)
            gt_cls = np.array([g[1] for g in rec["gt"]], dtype=np.int64)
            gt_xyxy = self._xywh_to_xyxy(gt_boxes)

            matched_gt = set()
            image_match_rows = []

            for c in np.unique(gt_cls):
                gt_count_by_class[int(c)] = gt_count_by_class.get(int(c), 0) + int((gt_cls == c).sum())

            order = np.argsort(-pred_scores)
            for p_idx in order:
                c = int(pred_cls[p_idx])
                same_cls = np.where(gt_cls == c)[0]
                best_iou = 0.0
                best_gt_idx = -1
                for gi in same_cls:
                    if gi in matched_gt:
                        continue
                    iou = self._iou_np(pred_xyxy[p_idx], gt_xyxy[gi])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = int(gi)

                is_tp = 1 if best_gt_idx >= 0 and best_iou >= 0.5 else 0
                if is_tp:
                    matched_gt.add(best_gt_idx)

                detections_by_class.setdefault(c, []).append((float(pred_scores[p_idx]), is_tp))
                image_match_rows.append({
                    "score": float(pred_scores[p_idx]),
                    "correct": int(is_tp),
                })

            per_image_matches.extend(image_match_rows)

        ap_per_class = {}
        all_tp = 0
        all_fp = 0
        all_gt = int(sum(gt_count_by_class.values()))

        for c, dets in detections_by_class.items():
            dets_sorted = sorted(dets, key=lambda x: x[0], reverse=True)
            tp = np.array([d[1] for d in dets_sorted], dtype=np.float32)
            fp = 1.0 - tp
            c_tp = np.cumsum(tp)
            c_fp = np.cumsum(fp)

            n_gt = max(gt_count_by_class.get(c, 0), 1)
            recall = c_tp / n_gt
            precision = c_tp / np.maximum(c_tp + c_fp, 1e-7)
            ap = self._compute_ap(recall, precision)
            ap_per_class[int(c)] = ap

            all_tp += int(tp.sum())
            all_fp += int(fp.sum())

        valid_aps = [v for k, v in ap_per_class.items() if gt_count_by_class.get(k, 0) > 0]
        map50 = float(np.mean(valid_aps)) if valid_aps else 0.0
        precision = all_tp / max(all_tp + all_fp, 1)
        recall = all_tp / max(all_gt, 1)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-7)

        return {
            "status": "ok",
            "dataset": f"COCO {split}2017",
            "images_evaluated": len(records),
            "iou_threshold": 0.5,
            "map50": round(map50, 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "per_class_ap_count": len(valid_aps),
            "calibration_rows": per_image_matches,
        }

    def _calibration_from_eval(self, eval_result: dict) -> dict:
        if eval_result.get("status") != "ok":
            return {
                "status": "skipped_no_eval_results",
                "message": "Evaluation did not produce detection matches.",
            }

        rows = eval_result.get("calibration_rows", [])
        if not rows:
            return {
                "status": "skipped_empty",
                "message": "No detections available for calibration computation.",
            }

        conf = torch.tensor([r["score"] for r in rows], dtype=torch.float32)
        corr = torch.tensor([r["correct"] for r in rows], dtype=torch.long)
        ece = CalibrationMetrics.expected_calibration_error(conf, corr)

        return {
            "status": "ok",
            "ece": round(float(ece), 4),
            "mean_confidence": round(float(conf.mean().item()), 4),
            "confidence_std": round(float(conf.std().item()), 4),
            "num_predictions": int(conf.numel()),
        }

    def _benchmark_adversarial_on_eval(self, data_root: str, split: str, max_images: int) -> dict:
        logger.info("Benchmark: adversarial evaluation on labeled data")

        baseline = self._evaluate_on_coco(data_root, split, max_images)
        if baseline.get("status") != "ok":
            return {
                "status": "skipped",
                "reason": baseline.get("message", "baseline evaluation failed"),
            }

        def perturb_eps(eps: float) -> Callable[[torch.Tensor], torch.Tensor]:
            def _fn(x: torch.Tensor) -> torch.Tensor:
                return (x + eps * torch.sign(torch.randn_like(x))).clamp(0.0, 1.0)
            return _fn

        eps_2 = self._evaluate_on_coco(data_root, split, max_images, perturb=perturb_eps(2 / 255))
        eps_4 = self._evaluate_on_coco(data_root, split, max_images, perturb=perturb_eps(4 / 255))

        if eps_2.get("status") != "ok" or eps_4.get("status") != "ok":
            return {
                "status": "partial",
                "baseline": baseline,
                "eps_2_255": eps_2,
                "eps_4_255": eps_4,
            }

        return {
            "status": "ok",
            "baseline_map50": baseline["map50"],
            "eps_2_255_map50": eps_2["map50"],
            "eps_4_255_map50": eps_4["map50"],
            "map50_drop_eps_2_255": round(baseline["map50"] - eps_2["map50"], 4),
            "map50_drop_eps_4_255": round(baseline["map50"] - eps_4["map50"], 4),
        }

    def print_summary(self):
        print("\n" + "=" * 72)
        print("  BENCHMARK REPORT")
        print("=" * 72)

        lat = self.results.get("latency", {})
        if lat:
            print("\nPerformance (synthetic)")
            print(f"  Latency: {lat.get('mean_ms', 0):.2f} ms | FPS: {lat.get('fps', 0):.1f}")

        qual = self.results.get("detection_quality", {})
        if qual.get("status") == "ok":
            print("\nDetection Quality (real data)")
            print(f"  mAP@0.5: {qual.get('map50', 0):.4f}")
            print(f"  Precision: {qual.get('precision', 0):.4f}")
            print(f"  Recall: {qual.get('recall', 0):.4f}")
            print(f"  F1: {qual.get('f1', 0):.4f}")
            print(f"  Images: {qual.get('images_evaluated', 0)}")
        else:
            print("\nDetection Quality")
            print(f"  {qual.get('message', 'Not available')}")

        cal = self.results.get("calibration", {})
        if cal.get("status") == "ok":
            print("\nCalibration (real data)")
            print(f"  ECE: {cal.get('ece', 0):.4f}")
            print(f"  Predictions: {cal.get('num_predictions', 0)}")

        adv = self.results.get("adversarial_eval", {})
        if adv.get("status") == "ok":
            print("\nAdversarial Quality (real data)")
            print(f"  Baseline mAP@0.5: {adv.get('baseline_map50', 0):.4f}")
            print(f"  eps=2/255 mAP@0.5: {adv.get('eps_2_255_map50', 0):.4f}")
            print(f"  eps=4/255 mAP@0.5: {adv.get('eps_4_255_map50', 0):.4f}")

        print("\n" + "=" * 72)
