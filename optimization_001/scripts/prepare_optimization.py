"""Freeze a new image-disjoint protocol before any optimization run starts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from swapgrasp.data import read_jsonl,write_jsonl

source=root.parent/'data/submission_001'
destination=root.parent/'data/optimization_001';destination.mkdir(exist_ok=False)
rows=read_jsonl(source/'manifest.jsonl');pairs=read_jsonl(source/'train_pairs.jsonl')
train_images=sorted({r['image_id'] for r in rows if r['split']=='train'})
paired={p['image_id'] for p in pairs if p['split']=='train'}
rng=random.Random(20261111)
holdout=set()
for group in ([i for i in train_images if i in paired],[i for i in train_images if i not in paired]):
    rng.shuffle(group);holdout.update(group[:round(len(group)*.12)])
old_val=sorted({r['image_id'] for r in rows if r['split']=='val'})
val_pair={p['image_id'] for p in pairs if p['split']=='val'}
rankfit=set()
for group in ([i for i in old_val if i in val_pair],[i for i in old_val if i not in val_pair]):
    rng.shuffle(group);rankfit.update(group[:len(group)//2])
result=[]
for original in rows:
    row=dict(original);old=row['split']
    if old=='test':continue
    row['split']=('test' if row['image_id'] in holdout else 'train') if old=='train' else (
        'rankfit' if row['image_id'] in rankfit else 'val')
    for key in ('image','mask','ignore_mask'):
        if row.get(key) and not Path(row[key]).is_absolute():row[key]=str((source/row[key]).resolve())
    result.append(row)
by_id={r['sample_id']:r for r in result};newpairs=[]
for original in pairs:
    if original['sample_a'] not in by_id or original['sample_b'] not in by_id:continue
    a,b=(by_id[original[k]] for k in ('sample_a','sample_b'))
    assert a['split']==b['split']
    newpairs.append({**original,'split':a['split']})
write_jsonl(destination/'manifest.jsonl',result);write_jsonl(destination/'pairs.jsonl',newpairs)
splits={s:{r['image_id'] for r in result if r['split']==s} for s in ('train','val','rankfit','test')}
assert sum(map(len,splits.values()))==len(set.union(*splits.values()))
protocol={'seed_split':20261111,'source_manifest_sha256':hashlib.sha256((source/'manifest.jsonl').read_bytes()).hexdigest(),
 'images':{s:len(ids) for s,ids in splits.items()},'queries':dict(Counter(r['split'] for r in result)),
 'pair_counts':dict(Counter(p['split'] for p in newpairs)),
 'policy':'Train all detectors from ImageNet initialization; new holdout was old training data, never used by these runs. Old official test excluded. rankfit fits frozen-detector candidate scorers only; val selects detector/scorer checkpoints; test opened once after training choices fixed.'}
(destination/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))
