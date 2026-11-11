"""Create fixed small overfit and pilot configs from an audited real manifest."""
import argparse
from pathlib import Path
import random
import sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import yaml
from swapgrasp.data import read_jsonl,write_jsonl

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True)
p.add_argument('--pairs',required=True);p.add_argument('--steps',type=int,default=300)
p.add_argument('--overfit-steps',type=int,default=1500)
p.add_argument('--overfit-image-prefix',action='append',default=[])
p.add_argument('--output',default='configs/generated');a=p.parse_args()
manifest=Path(a.manifest).resolve();pairs_path=Path(a.pairs).resolve()
rows=read_jsonl(manifest);pairs=read_jsonl(pairs_path)
train_pairs=[pair for pair in pairs if pair['split']=='train']
if not train_pairs:raise ValueError('Audited different-object training pairs are required for swap pilot')
chosen={row['image_id'] for row in rows if row['split']=='train' and any(row['image_id'].startswith(s) for s in a.overfit_image_prefix)}
if a.overfit_image_prefix and len(chosen)!=len(a.overfit_image_prefix):raise ValueError('Overfit prefixes must each select one distinct train image')
rng=random.Random(42)
pair_order=train_pairs.copy();rng.shuffle(pair_order)
if not chosen:
    for pair in pair_order:
        chosen.add(pair['image_id'])
        if len(chosen)>=4:break
if not any(pair['image_id'] in chosen for pair in train_pairs):raise ValueError('Overfit images must include a verified pair')
overfit_rows=[row for row in rows if row['split']!='train' or row['image_id'] in chosen]
overfit_manifest=manifest.parent/'overfit.jsonl';overfit_pairs=manifest.parent/'overfit_pairs.jsonl'
write_jsonl(overfit_manifest,overfit_rows)
ids={row['sample_id'] for row in overfit_rows}
write_jsonl(overfit_pairs,[pair for pair in pairs if pair['sample_a'] in ids and pair['sample_b'] in ids])
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
for name,weight in [('baseline',0.),('swap',.1),('overfit',.1)]:
    config=yaml.safe_load((root/'configs/swap.yaml').read_text())
    config['data'].update(root=str(manifest.parent),manifest=str(overfit_manifest if name=='overfit' else manifest),
                          pairs=str(overfit_pairs if name=='overfit' else pairs_path))
    config['loss']['lambda_swap']=weight
    config['train'].update(max_steps=a.overfit_steps if name=='overfit' else a.steps,warmup_steps=20,log_every=10,
                           validate_every=100,snapshot_every=100,amp_dtype='bfloat16')
    (out/f'{name}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
print('CONFIGS',out,'overfit training images',len(chosen),'all queries',len(rows),'train pairs',len(train_pairs))
