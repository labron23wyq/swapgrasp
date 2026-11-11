"""Compare an uninterrupted run with continuation from its step-2 checkpoint."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import yaml

root=Path(__file__).resolve().parents[1];os.chdir(root)
out=Path('runs')/f'resume_check_{int(time.time())}';out.mkdir(parents=True)
config=yaml.safe_load(Path('configs/synthetic.yaml').read_text())
config['train'].update(max_steps=4,validate_every=2,snapshot_every=2)
if '--prefetch' in sys.argv:config['train'].update(prefetch=True,cache_images=8,cache_targets=16)
cfg=out/'config.yaml';cfg.write_text(yaml.safe_dump(config))
for run,extra in [('full',[]),('resumed',['--resume',str(out/'full/step_2.pt')])]:
 subprocess.run([sys.executable,'train.py','--config',str(cfg),'--output',str(out/run),*extra],check=True)
a=torch.load(out/'full/last.pt',map_location='cpu',weights_only=False)
b=torch.load(out/'resumed/last.pt',map_location='cpu',weights_only=False)
assert a['step']==b['step']==4 and a['sampler']==b['sampler']
errors={k:float((v-b['model'][k]).abs().max()) for k,v in a['model'].items() if v.is_floating_point()}
assert max(errors.values())<1e-7, max(errors.items(),key=lambda x:x[1])
full=[json.loads(s) for s in (out/'full/schedule.jsonl').read_text().splitlines()]
resumed=[json.loads(s) for s in (out/'resumed/schedule.jsonl').read_text().splitlines()]
assert full[2:]==resumed
result={'passed':True,'max_model_parameter_difference':max(errors.values()),'same_sampler_schedule':True,'synthetic_only':True,'output':str(out)}
Path('artifacts/resume_check.json').write_text(json.dumps(result,indent=2));print(result)

