"""Run engineering checks and one synthetic GPU training/evaluation cycle."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1];os.chdir(root)
py=sys.executable
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=env.get('CUDA_VISIBLE_DEVICES','0')
env['HF_ENDPOINT']='https://hf-mirror.com'
out=f'runs/smoke_{int(time.time())}'
commands=[
 [py,'-m','pytest','-q'],
 [py,'tools.py','synthetic'],
 [py,'tools.py','audit','--manifest','data/synthetic/manifest.jsonl','--output','artifacts/synthetic_audit.json'],
 [py,'train.py','--config','configs/synthetic.yaml','--output',out],
 [py,'evaluate.py','predict','--checkpoint',out+'/best.pt','--manifest','data/synthetic/manifest.jsonl','--split','test','--output',out+'/predictions.jsonl','--visualize-dir',out+'/prediction_visuals'],
 [py,'evaluate.py','score','--manifest','data/synthetic/manifest.jsonl','--split','test','--predictions',out+'/predictions.jsonl','--pairs','data/synthetic/pairs.jsonl','--output',out+'/test'],
 [py,'visualize.py','--manifest','data/synthetic/manifest.jsonl','--predictions',out+'/predictions.jsonl','--output',out+'/gallery.html'],
 [py,'visualize.py','--log',out+'/metrics.jsonl','--output',out+'/curves.png'],
 [py,'explain.py','--checkpoint',out+'/best.pt','--baseline-checkpoint',out+'/last.pt',
  '--manifest','data/synthetic/manifest.jsonl','--pairs','data/synthetic/pairs.jsonl',
  '--predictions',out+'/predictions.jsonl','--output',out+'/explanations']]
results=[]
for command in commands:
 print('RUN',' '.join(command),flush=True)
 p=subprocess.run(command,env=env)
 results.append({'command':command,'exit_code':p.returncode})
 if p.returncode:
  break
Path('artifacts/engineering_verification.json').write_text(json.dumps({'synthetic_only':True,'run':out,'checks':results,'passed':all(r['exit_code']==0 for r in results)},indent=2))
raise SystemExit(0 if all(r['exit_code']==0 for r in results) else 1)
