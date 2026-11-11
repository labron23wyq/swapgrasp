"""Exercise real pretrained encoders and full-resolution AMP on fixture inputs."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
import time

root = Path(__file__).resolve().parents[1]
os.chdir(root)
sys.path.insert(0, str(root))
import torch
from swapgrasp.data import ManifestDataset
from swapgrasp.losses import grasp_loss, swap_loss
from swapgrasp.model import SwapGrasp, TextEncoder

torch.set_num_threads(4)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device('cuda:0')
started = time.monotonic()
dataset = ManifestDataset('data/synthetic/manifest.jsonl', 'train', 320)
batch = dataset.batch([0, 1])
encoder = TextEncoder(device, pretrained='weights/clip-vit-b32-openai.bin')
embeddings = encoder(batch['texts'])
assert embeddings.shape == (2, 512) and torch.isfinite(embeddings).all()
assert not embeddings.requires_grad and not torch.is_inference(embeddings)
assert torch.equal(embeddings, encoder(batch['texts']))
model = SwapGrasp(channels=128, image_weights_path='weights/resnet18-f37072fd.pth').to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype == torch.float16)
records = []
for step in range(2):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=amp_dtype):
        prediction = model(batch['images'].to(device), encoder(batch['texts']), batch['mapping'].to(device))
    assert prediction.shape == (2, 5, 320, 320)
    base, parts = grasp_loss(prediction, batch['targets'])
    swap = swap_loss(prediction[:, 0], [t['support'].to(device) for t in batch['targets']], [(0, 1)])
    loss = base + .1 * swap
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        broken = [name for name, parameter in model.named_parameters()
                  if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
        print({'finite_components': {name: float(value.detach()) for name, value in parts.items()},
               'swap': float(swap.detach()), 'loss': float(loss.detach()),
               'broken_gradient_parameters': broken,
               'prediction_finite': bool(torch.isfinite(prediction).all()),
               'prediction_minmax': [float(torch.nan_to_num(prediction).min()),
                                     float(torch.nan_to_num(prediction).max())]}, flush=True)
        raise FloatingPointError('Nonfinite AMP gradient norm')
    assert all(film[-1].weight.grad is not None for film in model.film)
    assert any(film[-1].weight.grad.abs().sum() > 0 for film in model.film)
    scaler.step(optimizer)
    scaler.update()
    records.append({'step': step + 1, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
    print(records[-1], flush=True)
torch.cuda.synchronize()
result = {
    'passed': True, 'synthetic_only': True, 'pretrained_encoders': True,
    'amp': True, 'amp_dtype': str(amp_dtype), 'deterministic_algorithms': True, 'image_size': 320,
    'output_shape': list(prediction.shape), 'steps': records,
    'peak_gpu_gib': torch.cuda.max_memory_allocated() / 2**30,
    'seconds': time.monotonic() - started,
}
Path('artifacts/pretrained_check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
