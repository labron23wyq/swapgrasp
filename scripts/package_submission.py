"""Prepare reviewable source/results ZIPs without publishing data or weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];run=Path(a.run).resolve();output=root/'artifacts'/run.name
output.mkdir(parents=True,exist_ok=True)
protocol=json.loads((run/'protocol.json').read_text())
workflow=json.loads((run/'workflow_state.json').read_text())
evidence=json.loads((run/'evidence.json').read_text())
run_names=set(protocol['runs'])
if workflow['status']!='complete' or set(workflow['evaluated'])!=run_names or set(evidence['test_metrics'])!=run_names:
    raise ValueError('Refusing to package incomplete evaluation as final')
if not all(evidence['same_schedule'].values()):
    raise ValueError('Matched runs do not share the same sample schedule')
paper=root/'paper';provenance=json.loads((paper/'result_provenance.json').read_text())
if provenance['page_count']>2 or provenance.get('evidence_sha256')!=hashlib.sha256((run/'evidence.json').read_bytes()).hexdigest():
    raise ValueError('The final two-page PDF must be built from this evidence')

def finish_zip(path):
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist())!=len(set(archive.namelist())) or archive.testzip() is not None:
            raise ValueError(f'Invalid ZIP: {path}')
    path.replace(path.with_suffix(''))

files={}
source=output/'source.zip';source_temp=source.with_suffix('.zip.tmp')
with zipfile.ZipFile(source_temp,'w',zipfile.ZIP_DEFLATED) as archive:
    for folder in (root,root/'swapgrasp',root/'scripts',root/'tests',root/'configs',root/'paper'):
        for path in folder.iterdir():
            if not path.is_file() or (path.name!='.gitignore' and path.suffix not in ('.py','.toml','.md','.yaml','.tex','.sty','.bst','.bib')):continue
            name=path.relative_to(root).as_posix();archive.write(path,name)
            files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    requirements=root/'artifacts/requirements-resolved.txt'
    if requirements.is_file():archive.write(requirements,'artifacts/'+requirements.name)
    for name in sorted(run_names):
        path=run/name/'config.yaml'
        archive.write(path,'configs/completed_'+name+'.yaml')
finish_zip(source_temp)
(output/'source_sha256.json').write_text(json.dumps(files,indent=2))
results=output/'results.zip';results_temp=results.with_suffix('.zip.tmp')
root_files={'REPORT.zh-CN.md','REPORT.zh-CN.html','protocol.json','workflow_state.json','data_audit.json',
            'evidence.json','comparisons.json','live_status.json','live_training.png','visual_manifest.jsonl',
            'threshold_curves.png','threshold_curves.pdf','geometry_cdf.png','geometry_cdf.pdf',
            'batch_benchmark_full.json'}
run_files={'config.yaml','environment.json','metrics.jsonl','train.log','curves.png','curves.svg',
           'val_predictions.jsonl','test_predictions.jsonl','train_visual_predictions.jsonl',
           'test_visual_predictions.jsonl'}
allowed={'.json','.jsonl','.png','.svg','.pdf','.html','.md','.csv','.yaml','.txt','.log'}
with zipfile.ZipFile(results_temp,'w',zipfile.ZIP_DEFLATED) as archive:
    for path in run.rglob('*'):
        if not path.is_file():continue
        parts=path.relative_to(run).parts
        keep=(len(parts)==1 and parts[0] in root_files) or parts[0] in {'explanations','inference_demo'}
        if parts[0] in run_names and len(parts)>1:
            keep=parts[1] in (run_files if len(parts)==2 else {'val','test','train_visuals','test_visuals'})
            if len(parts)==3 and parts[1]=='tensorboard' and parts[2].startswith('events.out.tfevents.'):
                archive.write(path,Path(*parts).as_posix())
        if keep and path.suffix in allowed:archive.write(path,Path(*parts).as_posix())
    manifest=Path(protocol['manifest'])
    for name in ('manifest.jsonl','train_pairs.jsonl','protocol.json','adapter_summary.json','quarantine.jsonl'):
        path=manifest.parent/name
        if path.is_file():archive.write(path,'provenance/'+name)
    catalog=root/'data/downloads/catalog.json'
    if catalog.is_file():archive.write(catalog,'provenance/downloads/catalog.json')
    for path in (root/'data/downloads').glob('*/*.verified.json'):
        archive.write(path,'provenance/downloads/'+path.relative_to(root/'data/downloads').as_posix())
    for path in (root/'weights').glob('*.verified.json'):
        archive.write(path,'provenance/weights/'+path.name)
    for path in paper.iterdir():
        if path.is_file() and path.suffix in ('.pdf','.tex','.bib','.sty','.bst','.json'):
            archive.write(path,'paper/'+path.name)
finish_zip(results_temp)
digests={}
for path in (source,results):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    digests[path.name]=digest.hexdigest()
(output/'sha256.json').write_text(json.dumps(digests,indent=2))
print('PACKAGED',source,source.stat().st_size,results,results.stat().st_size,flush=True)
