"""Launch two seeds for the fixed detector ablations without changing old runs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
root=Path(__file__).resolve().parents[1]
run=root.parent/'runs/optimization_001';run.mkdir(exist_ok=False)
data=root.parent/'data/optimization_001'
state={'status':'training','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'manifest':str(data/'manifest.jsonl'),'pairs':str(data/'pairs.jsonl'),
       'max_steps':3500,'seeds':[42,123],'runs':{},'protocol':json.loads((data/'protocol.json').read_text()),
       'fixed_methods':{'original':'original query-weighted sampling, 50% pair queries, no swap',
          'balanced':'image-balanced sampling, 12.5% pair queries, no swap',
          'hard_swap':'balanced + top-quarter competing support hinge ranking, start 500 / ramp 500, lambda .05'}}
for seed_i,seed in enumerate(state['seeds']):
    for method_i,method in enumerate(('original','balanced','hard_swap')):
        gpu=seed_i*3+method_i;name=f'{method}_s{seed}';folder=run/name;folder.mkdir()
        cfg=yaml.safe_load((root.parent/'runs/submission_001/baseline_s42/config.yaml').read_text())
        cfg['seed']=seed;cfg['data'].update(root=str(data),manifest=state['manifest'],pairs=state['pairs'])
        cfg['sampler'].update(queries=256,pairs_per_step=64 if method=='original' else 16,
                              image_balanced=method!='original')
        cfg['model']['image_weights_path']=str(root.parent/'weights/resnet18-f37072fd.pth')
        cfg['model']['text_pretrained']=str(root.parent/'weights/clip-vit-b32-openai.bin')
        cfg['loss'].update(lambda_swap=.05 if method=='hard_swap' else 0.,swap_type='hard' if method=='hard_swap' else 'original',
                           swap_start=500,swap_ramp=500,margin=.5 if method=='hard_swap' else .2,top_fraction=.25)
        cfg['train'].update(max_steps=state['max_steps'],validate_every=250,snapshot_every=3500,
                            cache_images=1024,cache_targets=4096)
        config=folder/'config.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),SWAPGRASP_PHYSICAL_GPU=str(gpu),
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
        with (folder/'train.log').open('xb') as log:
            worker=subprocess.Popen([sys.executable,'-u','train.py','--config',str(config),'--output',str(folder)],
                cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        state['runs'][name]={'gpu':gpu,'pid':worker.pid,'seed':seed,'method':method,'directory':str(folder)}
        (run/'protocol.json').write_text(json.dumps(state,indent=2));print('STARTED',name,gpu,worker.pid,flush=True)
