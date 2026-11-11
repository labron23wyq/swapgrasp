"""Check real-data optimization on explicitly inspected training images.

This can run without swap pairs. It measures training-set fitting, not
generalization or the benefit of the proposed swap loss.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root));os.chdir(root)
import yaml
from swapgrasp.data import read_jsonl,write_jsonl
from tools import audit

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
p.add_argument('--image-prefix',action='append',required=True)
p.add_argument('--steps',type=int,default=300);p.add_argument('--output',required=True)
a=p.parse_args();source=Path(a.manifest).resolve();out=Path(a.output).resolve()
out.mkdir(parents=True,exist_ok=True)
if (out/'state.json').exists():raise FileExistsError('Choose a new overfit run')
rows=read_jsonl(source)
chosen={r['image_id'] for r in rows if r['split']=='train' and any(r['image_id'].startswith(s) for s in a.image_prefix)}
if len(chosen)!=len(a.image_prefix):raise ValueError('Image prefixes must each select one distinct training image')
selected=[dict(r) for r in rows if r['split']=='val' or (r['split']=='train' and r['image_id'] in chosen)]
for row in selected:
    for key in ['image','mask','ignore_mask']:
        if row.get(key):row[key]=str((source.parent/row[key]).resolve())
manifest=out/'manifest.jsonl';write_jsonl(manifest,selected)
if not audit(manifest,None,out/'data_audit.json'):raise ValueError('Data audit failed')
config=yaml.safe_load((root/'configs/baseline.yaml').read_text())
config['data'].update(root=str(source.parent),manifest=str(manifest),pairs=None)
config['sampler']['pairs_per_step']=0;config['loss']['lambda_swap']=0.
config['train'].update(max_steps=a.steps,warmup_steps=20,log_every=10,validate_every=100,snapshot_every=100,amp_dtype='bfloat16')
config_file=out/'config_input.yaml';config_file.write_text(yaml.safe_dump(config,sort_keys=False))
state={'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'scope':'real_data_optimization_check_only_no_generalization_claim','training_images':sorted(chosen),
       'synthetic_only':False,'checks':[]}
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=env.get('CUDA_VISIBLE_DEVICES','0')
def run(name,command):
    state['active']=name;(out/'state.json').write_text(json.dumps(state,indent=2))
    print('RUN',name,flush=True)
    with open(out/(name+'.log'),'w') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    state['checks'].append({'name':name,'exit_code':result.returncode})
    if result.returncode:
        state['status']='failed';(out/'state.json').write_text(json.dumps(state,indent=2))
        raise SystemExit(result.returncode)

run('train',[sys.executable,'train.py','--config',str(config_file),'--output',str(out/'model')])
run('predict_train',[sys.executable,'evaluate.py','predict','--checkpoint',str(out/'model/last.pt'),
    '--manifest',str(manifest),'--split','train','--output',str(out/'train_predictions.jsonl'),
    '--visualize-dir',str(out/'prediction_visuals')])
run('score_train',[sys.executable,'evaluate.py','score','--manifest',str(manifest),'--split','train',
    '--predictions',str(out/'train_predictions.jsonl'),'--output',str(out/'train_scores')])
run('gallery',[sys.executable,'visualize.py','--manifest',str(manifest),'--split','train',
    '--predictions',str(out/'train_predictions.jsonl'),'--output',str(out/'gallery.html')])
run('curves',[sys.executable,'visualize.py','--log',str(out/'model/metrics.jsonl'),'--output',str(out/'curves.png')])
metrics=json.loads((out/'train_scores/metrics.json').read_text())
state.update(status='complete',active=None,training_metrics=metrics)
(out/'state.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2),flush=True)
