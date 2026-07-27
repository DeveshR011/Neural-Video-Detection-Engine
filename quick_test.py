"""Quick smoke test for core model/security plumbing.

This script is intentionally lightweight and should complete quickly.
"""

import logging
import os
import torch

from model.detector import SecureYOLO
from model.watermark import WatermarkEmbedder, WatermarkVerifier
from security.runtime_integrity import LayerHashRegistry

logging.basicConfig(level=logging.INFO, format="%(name)-18s | %(message)s")


def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SecureYOLO(80, fpn_channels=128).to(device).eval()

	params = sum(p.numel() for p in model.parameters())
	print(f"model_params={params}")

	x = torch.rand(1, 3, 640, 640, device=device)
	with torch.no_grad():
		box_dist, cls_logits, anchor_points, strides = model(x)
		pred = model.predict_padded(x)
	print(f"box_dist_shape={tuple(box_dist.shape)}")
	print(f"cls_logits_shape={tuple(cls_logits.shape)}")
	print(f"anchors={anchor_points.shape[0]}")
	print(f"pred_shape={tuple(pred.shape)}")

	wk = os.urandom(32)
	model = WatermarkEmbedder(wk).embed(model, strength=0.001)
	valid, acc = WatermarkVerifier(wk).verify(model)
	print(f"watermark_valid={valid} watermark_acc={acc:.4f}")

	reg = LayerHashRegistry()
	reg.register_model(model)
	print(f"integrity_layers={len(reg.get_registered_names())}")

	print("quick_test_status=ok")


if __name__ == "__main__":
	main()
