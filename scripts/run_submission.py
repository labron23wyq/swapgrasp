"""Run matched multi-seed Baseline/SwapGrasp trials, then score every test query."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root))
from swapgrasp.data import read_jsonl,write_jsonl
from tools import audit

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--pairs',required=True)
p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=256)
p.add_argument('--epochs',type=int,default=50);p.add_argument('--seeds',nargs='+',type=int,default=[42,123,3407])
p.add_argument('--gpus',nargs='+',type=int,default=[0,1,2,3,4,5])
a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
manifest=Path(a.manifest).resolve();pairs_path=Path(a.pairs).resolve();rows=read_jsonl(manifest);pairs=read_jsonl(pairs_path)
if len(a.gpus)!=2*len(a.seeds):raise ValueError('Need one GPU per model in every seed pair')
if a.batch_size<4 or a.batch_size%4:raise ValueError('Batch size must be divisible by four for paired sampling')
if not audit(manifest,None,out/'data_audit.json'):raise ValueError('Dataset audit failed')
train_rows=[r for r in rows if r['split']=='train'];steps_per_epoch=math.ceil(len(train_rows)/a.batch_size)
max_steps=steps_per_epoch*a.epochs
protocol={'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
          'manifest':str(manifest),'pairs':str(pairs_path),'queries':len(rows),
          'images':len({r['image_id'] for r in rows}),
          'images_by_split':{s:len({r['image_id'] for r in rows if r['split']==s}) for s in ('train','val','test')},
          'queries_by_split':dict(Counter(r['split'] for r in rows)),
          'pairs_by_split':dict(Counter(q['split'] for q in pairs)),
          'batch_size_queries':a.batch_size,'pairs_per_step':a.batch_size//4,
          'epochs_equivalent':a.epochs,'steps_per_epoch':steps_per_epoch,'max_steps':max_steps,
          'seeds':a.seeds,'gpus':a.gpus,'runs':{},
          'scope':'image-disjoint multi-object diagnostic subset; not the official full benchmark',
          'test_policy':'Test is evaluated only after checkpoint selection by validation Top-1.'}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for seed_i,seed in enumerate(a.seeds):
    for method_i,method in enumerate(('baseline','swap')):
        run_name=f'{method}_s{seed}';gpu=a.gpus[2*seed_i+method_i]
        run=out/run_name
        if run.exists() and any(run.iterdir()):
            raise FileExistsError(f'Output already has files: {run}')
        run.mkdir(parents=True,exist_ok=True);cfg=yaml.safe_load((root/'configs/swap.yaml').read_text())
        cfg['seed']=seed;cfg['data'].update(root=str(manifest.parent),manifest=str(manifest),pairs=str(pairs_path))
        cfg['sampler'].update(queries=a.batch_size,pairs_per_step=a.batch_size//4)
        cfg['loss']['lambda_swap']=0. if method=='baseline' else .1
        cfg['train'].update(max_steps=max_steps,warmup_steps=min(100,max_steps//10),log_every=25,
                            validate_every=max(steps_per_epoch,250),snapshot_every=max(steps_per_epoch*5,500),
                            prefetch=True,cache_images=2048,cache_targets=8192,amp_dtype='bfloat16')
        config=run/'config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        env['SWAPGRASP_PHYSICAL_GPU']=str(gpu)
        log=(run/'train.log').open('xb')
        worker=subprocess.Popen([sys.executable,'-u','train.py','--config',str(config),'--output',str(run)],
                                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,start_new_session=True)
        log.close();protocol['runs'][run_name]={'gpu':gpu,'seed':seed,'method':method,'pid':worker.pid,
                                                'status':'running','directory':str(run)}
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
        print('STARTED',run_name,'GPU',gpu,'PID',worker.pid,'steps',max_steps,
              'epochs',a.epochs,'queries/step',a.batch_size,flush=True)
print(json.dumps(protocol,indent=2),flush=True)
