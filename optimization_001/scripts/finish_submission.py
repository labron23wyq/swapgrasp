"""Monitor detached training, evaluate fixed best checkpoints, and build final evidence."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parents[1];os.chdir(root);sys.path.insert(0,str(root))
from swapgrasp.data import read_jsonl,write_jsonl
from swapgrasp.metrics import paired_bootstrap


def alive(pid):
    path=Path('/proc')/str(pid)/'stat'
    return path.exists() and path.read_text().split()[2]!='Z'


def run_command(name,command,out,env):
    log=out/(name+'.log');print('RUN',name,flush=True)
    with log.open('ab') as stream:
        result=subprocess.run(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{name} failed ({result.returncode}); see {log}')


def evaluate_run(item,manifest,pairs,run,env):
    folder=Path(item['directory']);name=folder.name
    for split in ('val','test'):
        output=folder/f'{split}_predictions.jsonl'
        if not output.exists():
            run_command(name+'_'+split+'_predict',[sys.executable,'evaluate.py','predict','--checkpoint',str(folder/'best.pt'),
                        '--manifest',str(manifest),'--split',split,'--output',str(output)],run,env)
        run_command(name+'_'+split+'_score',[sys.executable,'evaluate.py','score','--manifest',str(manifest),
                    '--pairs',str(pairs),'--split',split,'--predictions',str(output),'--output',str(folder/split)],run,env)
    # Select visual examples by deterministic manifest order, before inspecting test successes.
    rows=read_jsonl(manifest);train=[r for r in rows if r['split']=='train'][:256]
    tests=[r for r in rows if r['split']=='test'][:48]
    subset=run/'visual_manifest.jsonl';write_jsonl(subset,train+tests)
    for split in ('train','test'):
        if item['seed']!=42 and split=='train':continue
        run_command(name+'_'+split+'_visuals',[sys.executable,'evaluate.py','predict','--checkpoint',str(folder/'best.pt'),
                    '--manifest',str(subset),'--root',str(manifest.parent),'--split',split,
                    '--output',str(folder/f'{split}_visual_predictions.jsonl'),
                    '--visualize-dir',str(folder/f'{split}_visuals')],run,env)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--eval-gpu',default='6');a=p.parse_args()
    run=Path(a.run).resolve();protocol=json.loads((run/'protocol.json').read_text())
    manifest=Path(protocol['manifest']);pairs=Path(protocol['pairs'])
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.eval_gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    state={'status':'monitoring_training','evaluated':[],'failed':{},'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    done=set()
    while len(done)<len(protocol['runs']):
        for name,item in protocol['runs'].items():
            if name in done:continue
            if alive(item['pid']):continue
            done.add(name);folder=Path(item['directory'])
            text=(folder/'train.log').read_text(errors='replace')
            if 'FINISHED '+str(protocol['max_steps'])+' steps;' not in text:
                state['failed'][name]=text[-2000:];continue
            try:
                state['status']='evaluating_'+name
                (run/'workflow_state.json').write_text(json.dumps(state,indent=2))
                evaluate_run(item,manifest,pairs,run,env);state['evaluated'].append(name)
            except Exception as e:
                state['failed'][name]=str(e)
        subprocess.run([sys.executable,'scripts/monitor_submission.py',str(run)],cwd=root,env=env)
        for name,item in protocol['runs'].items():
            if (Path(item['directory'])/'metrics.jsonl').exists():
                subprocess.run([sys.executable,'visualize.py','--log',str(Path(item['directory'])/'metrics.jsonl'),
                                '--output',str(Path(item['directory'])/'curves.png')],cwd=root,env=env,
                               stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        state['status']='monitoring_training' if len(done)<len(protocol['runs']) else 'finalizing'
        (run/'workflow_state.json').write_text(json.dumps(state,indent=2))
        if len(done)<len(protocol['runs']):time.sleep(45)
    comparison={}
    for seed in protocol['seeds']:
        baseline=run/f'baseline_s{seed}';swap=run/f'swap_s{seed}'
        if not all((d/'test/detailed_per_query.jsonl').exists() for d in (baseline,swap)):continue
        ba=read_jsonl(baseline/'test/detailed_per_query.jsonl');sw=read_jsonl(swap/'test/detailed_per_query.jsonl')
        comparison[str(seed)]={metric:paired_bootstrap(ba,sw,metric=metric)
                               for metric in ('grasp_at1','strict_at1','very_strict_at1')}
        if (baseline/'schedule.jsonl').read_bytes()!=(swap/'schedule.jsonl').read_bytes():
            state['failed'][f'schedule_s{seed}']='Baseline and swap schedules differ'
    (run/'comparisons.json').write_text(json.dumps(comparison,indent=2))
    if (run/'swap_s42/test_predictions.jsonl').exists():
        run_command('explanations',[sys.executable,'explain.py','--checkpoint',str(run/'swap_s42/best.pt'),
                    '--baseline-checkpoint',str(run/'baseline_s42/best.pt'),'--manifest',str(manifest),
                    '--pairs',str(pairs),'--predictions',str(run/'swap_s42/test_predictions.jsonl'),
                    '--output',str(run/'explanations'),'--limit','12'],run,env)
    report=root/'scripts/report_submission.py'
    if report.exists():run_command('report',[sys.executable,str(report),'--run',str(run)],run,env)
    state['status']='complete' if not state['failed'] else 'complete_with_failures'
    state['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (run/'workflow_state.json').write_text(json.dumps(state,indent=2));print(json.dumps(state,indent=2),flush=True)


if __name__=='__main__':main()
