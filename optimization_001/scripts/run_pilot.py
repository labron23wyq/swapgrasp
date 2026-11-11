"""Run an audited, visually inspected small real-data experiment sequence.

This is an engineering/diagnostic pilot, not an official benchmark result.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root))
from swapgrasp.data import read_jsonl
from tools import audit

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
p.add_argument('--pairs',required=True);p.add_argument('--output',required=True)
p.add_argument('--steps',type=int,default=300)
p.add_argument('--overfit-min-at1',type=float,default=.8)
p.add_argument('--overfit-steps',type=int,default=1500)
p.add_argument('--overfit-image-prefix',action='append',default=[])
a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
manifest=Path(a.manifest).resolve();pairs=Path(a.pairs).resolve()
if (out/'state.json').exists():raise FileExistsError('Use a new pilot output directory')
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=env.get('CUDA_VISIBLE_DEVICES','0')
if not audit(manifest,None,out/'data_audit.json'):raise ValueError('Pilot data audit failed')
state={'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
       'manifest':str(manifest),'synthetic_only':False,
       'scope':'small_image_disjoint_diagnostic_pilot_not_official_benchmark',
       'checks':[]}

def run(name,command):
    state['active']=name;(out/'state.json').write_text(json.dumps(state,indent=2))
    print('RUN',name,flush=True)
    with open(out/(name.replace('/','_')+'.log'),'w') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    state['checks'].append({'name':name,'exit_code':result.returncode})
    if result.returncode:
        state['status']='failed';(out/'state.json').write_text(json.dumps(state,indent=2))
        raise SystemExit(result.returncode)

configs=out/'configs'
build=[sys.executable,'scripts/build_pilot.py','--manifest',str(manifest),'--pairs',str(pairs),
       '--steps',str(a.steps),'--overfit-steps',str(a.overfit_steps),'--output',str(configs)]
for prefix in a.overfit_image_prefix:build.extend(['--overfit-image-prefix',prefix])
run('build_configs',build)
for method in ['overfit','baseline','swap']:
    import yaml
    config=yaml.safe_load((configs/f'{method}.yaml').read_text())
    method_manifest=config['data']['manifest'];method_pairs=config['data']['pairs'];folder=out/method
    run(method+'/train',[sys.executable,'train.py','--config',str(configs/f'{method}.yaml'),'--output',str(folder)])
    checkpoint=folder/('last.pt' if method=='overfit' else 'best.pt')
    if method=='overfit':
        run(method+'/train_predict',[sys.executable,'evaluate.py','predict','--checkpoint',str(checkpoint),
            '--manifest',method_manifest,'--split','train','--output',str(folder/'train_predictions.jsonl'),
            '--visualize-dir',str(folder/'train_visuals')])
        run(method+'/train_score',[sys.executable,'evaluate.py','score','--manifest',method_manifest,
            '--split','train','--predictions',str(folder/'train_predictions.jsonl'),'--pairs',method_pairs,
            '--output',str(folder/'train')])
        training_scores=json.loads((folder/'train/metrics.json').read_text())
        if training_scores['grasp_at1']<a.overfit_min_at1:
            state.update(status='overfit_gate_failed',active=None,overfit_scores=training_scores,
                         required_training_at1=a.overfit_min_at1)
            (out/'state.json').write_text(json.dumps(state,indent=2))
            print('Overfit gate failed; inspect training predictions before baseline/swap comparison.',flush=True)
            raise SystemExit(2)
    for split in ['train','val','test']:
        if method=='overfit' and split=='train':continue
        pred=folder/f'{split}_predictions.jsonl'
        run(method+'/'+split+'_predict',[sys.executable,'evaluate.py','predict','--checkpoint',str(checkpoint),
            '--manifest',method_manifest,'--split',split,'--output',str(pred),
            '--visualize-dir',str(folder/f'{split}_visuals')])
        run(method+'/'+split+'_score',[sys.executable,'evaluate.py','score','--manifest',method_manifest,
            '--split',split,'--predictions',str(pred),'--pairs',method_pairs,'--output',str(folder/split)])
    run(method+'/curves',[sys.executable,'visualize.py','--log',str(folder/'metrics.jsonl'),'--output',str(folder/'curves.png')])
run('comparison',[sys.executable,'evaluate.py','compare','--scores-a',str(out/'baseline/test/per_query.jsonl'),
    '--scores-b',str(out/'swap/test/per_query.jsonl'),'--output',str(out/'comparison.json')])
run('explanations',[sys.executable,'explain.py','--checkpoint',str(out/'swap/best.pt'),
    '--baseline-checkpoint',str(out/'baseline/best.pt'),'--manifest',str(manifest),'--pairs',str(pairs),
    '--predictions',str(out/'swap/test_predictions.jsonl'),'--output',str(out/'explanations')])
schedule_a=(out/'baseline/schedule.jsonl').read_text()
schedule_b=(out/'swap/schedule.jsonl').read_text()
if schedule_a!=schedule_b:raise ValueError('Baseline/method sample schedules differ')
table=[]
for method in ['overfit','baseline','swap']:
    scores={split:json.loads((out/method/split/'metrics.json').read_text()) for split in ['train','val','test']}
    table.append(f'| {method} | '+ ' | '.join(f'{scores[s]["grasp_at1"]:.3f} / {scores[s]["grasp_at5"]:.3f}' for s in scores)+' |')
report='''# Small real-data diagnostic pilot

This pilot checks optimization, data alignment and prompt conditioning. It is not an official protocol reproduction or evidence of a general improvement. Prefix records, if any, carry member-CRC-only provenance until the full archives are verified. Test results must not be used to tune hyperparameters.

| Model | Train @1 / @5 | Validation @1 / @5 | Test @1 / @5 |
|---|---|---|---|
'''+ '\n'.join(table)+'\n\nBaseline and swap used exactly the same sample schedule. See comparison.json for an image-cluster bootstrap interval; the small diagnostic sample limits interpretation.\n'
(out/'REPORT.md').write_text(report)
state.update(status='complete',active=None,same_baseline_swap_schedule=True)
(out/'state.json').write_text(json.dumps(state,indent=2));print(report,flush=True)
