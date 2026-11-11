"""Measure real-data training throughput and peak CUDA memory before fixing batch size."""
import argparse
import gc
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import random
import sys
import time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
import torch
import yaml
from swapgrasp.data import ManifestDataset,PairSampler
from swapgrasp.model import SwapGrasp,TextEncoder
from swapgrasp.losses import grasp_loss,swap_loss

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--pairs',required=True)
p.add_argument('--output',required=True);p.add_argument('--sizes',nargs='+',type=int,default=[8,16,32,48,64])
p.add_argument('--steps',type=int,default=6);a=p.parse_args()
torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True)
ds=ManifestDataset(a.manifest,'train',320)
results=[]
for batch in a.sizes:
    model=text=optimizer=output=loss=data=None
    try:
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        sampler=PairSampler(ds,a.pairs,queries=batch,pairs_per_step=batch//4,seed=42)
        cfg=yaml.safe_load((root/'configs/swap.yaml').read_text())
        model=SwapGrasp(128,True,image_weights_path=cfg['model']['image_weights_path']).cuda()
        text=TextEncoder(torch.device('cuda'),pretrained=cfg['model']['text_pretrained'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
        torch.cuda.reset_peak_memory_stats();times=[]
        for step in range(a.steps+2):
            torch.cuda.synchronize();t=time.monotonic();ids,pairs=sampler.next();data=ds.batch(ids)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(data['images'].cuda(),text(data['texts']),data['mapping'].cuda())
            base,_=grasp_loss(output,data['targets'])
            loss=base+.1*swap_loss(output[:,0],[t['support'].cuda() for t in data['targets']],pairs)
            optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            torch.cuda.synchronize()
            if step>=2:times.append(time.monotonic()-t)
        item={'batch_queries':batch,'pairs_per_step':batch//4,'seconds_per_step':float(np.median(times)),
              'queries_per_second':float(batch/np.median(times)),
              'peak_allocated_gb':torch.cuda.max_memory_allocated()/2**30,
              'peak_reserved_gb':torch.cuda.max_memory_reserved()/2**30,
              'total_gb':torch.cuda.get_device_properties(0).total_memory/2**30,'status':'ok'}
    except torch.cuda.OutOfMemoryError:
        item={'batch_queries':batch,'status':'out_of_memory'}
    results.append(item);print(json.dumps(item),flush=True)
    model=text=optimizer=output=loss=data=None;gc.collect();torch.cuda.empty_cache()
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(results,indent=2))
    if item['status']!='ok':break
