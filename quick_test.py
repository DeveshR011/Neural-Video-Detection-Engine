"""Quick verification test."""
import torch, logging, os
logging.basicConfig(level=logging.INFO, format='%(name)-18s | %(message)s')

from model.detector import SecureYOLO
from model.watermark import WatermarkEmbedder, WatermarkVerifier
from security.architecture_morph import ArchitectureMorpher
from security.runtime_integrity import LayerHashRegistry
from benchmarks.benchmark import BenchmarkSuite

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SecureYOLO(80, fpn_channels=128).to(device)
params = sum(p.numel() for p in model.parameters())
print(f'\n=== MODEL: {params:,} params ===')

wk = os.urandom(32)
model = WatermarkEmbedder(wk).embed(model, strength=0.001)
v, acc = WatermarkVerifier(wk).verify(model)
print(f'Watermark: valid={v}, accuracy={acc:.2%}')

model = ArchitectureMorpher().morph(model)

reg = LayerHashRegistry()
reg.register_model(model)
print(f'Integrity: {len(reg.get_registered_names())} layers')

bench = BenchmarkSuite(model, device)
results = bench.run_all(num_frames=30, output_path='secure_enclave/ml_benchmark_results.json')
bench.print_summary()
print('\nVERIFICATION COMPLETE')
