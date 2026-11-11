"""Data preparation, auditing and conservative pair construction."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import zipfile
import numpy as np
from PIL import Image,ImageDraw
from swapgrasp.data import ManifestDataset,build_pairs,read_jsonl,write_jsonl,load_target_mask


def synthetic(root):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    rng=random.Random(123); rows=[]
    for i in range(8):
        split='train' if i<4 else ('val' if i<6 else 'test')
        image_id=f'fixture_{i:03d}'; im=Image.new('RGB',(64,64),(35+i*3,40,48));draw=ImageDraw.Draw(im)
        centers=[(rng.randint(13,22),rng.randint(16,47)),(rng.randint(42,51),rng.randint(16,47))]
        for j,((x,y),color) in enumerate(zip(centers,[(210,50,45),(40,100,220)])):
            box=(x-8,y-5,x+8,y+5); draw.rectangle(box,fill=color)
            mask=Image.new('L',im.size); ImageDraw.Draw(mask).rectangle(box,fill=255)
            maskname=f'{image_id}_{j}.png'; mask.save(root/maskname)
            rows.append({'sample_id':f'{image_id}_{j}','image_id':image_id,'image':f'{image_id}.png',
                         'text':f'grasp the {"red" if j==0 else "blue"} object','object_id':j,'part_id':None,
                         'grasps':[[float(x),float(y),16.,10.,0.]],'angle_unit':'radian',
                         'split':split,'mask':maskname,'mask_level':'object','semantic_verified':True,
                         'source':'synthetic_engineering_fixture'})
        im.save(root/f'{image_id}.png')
    write_jsonl(root/'manifest.jsonl',rows)
    pairs=[]
    for split in ['train','val','test']:
        ds=ManifestDataset(root/'manifest.jsonl',split,64)
        ps,_=build_pairs(ds); pairs.extend(ps)
    write_jsonl(root/'pairs.jsonl',pairs)
    print('SYNTHETIC ENGINEERING FIXTURE:',len(rows),'queries',len(pairs),'pairs. Not research evidence.')


def audit(manifest,root,output):
    rows=read_jsonl(manifest); by_image=defaultdict(list); errors=[]
    ds=ManifestDataset(manifest,rows[0]['split'],64,root)
    ids=[r['sample_id'] for r in rows]
    if len(ids)!=len(set(ids)): errors.append('Duplicate sample_id')
    hashes=defaultdict(set)
    for row in rows:
        try:
            if not row['text'].strip(): raise ValueError('Empty instruction')
            if row.get('angle_unit')!='radian': raise ValueError('Expected radian canonical geometry')
            if not row['grasps']: raise ValueError('Empty GT')
            path=ds.path(row['image'])
            with Image.open(path) as im:
                w,h=im.size; im.verify()
            image_key=(row['image_id'],str(path.resolve()))
            by_image[image_key].append(row['split'])
            gs=np.asarray(row['grasps'],dtype=float)
            if gs.ndim!=2 or gs.shape[1]!=5 or not np.isfinite(gs).all(): raise ValueError('Invalid grasp vectors')
            if np.any(gs[:,2:4]<=0): raise ValueError('Nonpositive dimensions')
            if np.any(gs[:,:2]<0) or np.any(gs[:,0]>=w) or np.any(gs[:,1]>=h): raise ValueError('GT centre out of image')
            if row.get('mask'):
                mask=load_target_mask(row,ds.path(row['mask']))
                if mask.shape!=(h,w) or not mask.any(): raise ValueError('Invalid target mask')
        except (ValueError,KeyError,OSError) as e:
            errors.append({'sample_id':row.get('sample_id'),'error':str(e)})
    image_splits=defaultdict(set)
    for (iid,path),splits in by_image.items():
        image_splits[iid].update(splits)
        digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        hashes[digest].update(splits)
    for iid,splits in image_splits.items():
        if len(splits)>1: errors.append(f'Image leakage: {iid}')
    for digest,splits in hashes.items():
        if len(splits)>1: errors.append(f'Exact-image-hash leakage: {digest}')
    report={'queries':len(rows),'unique_images':len(image_splits),'splits':dict(Counter(r['split'] for r in rows)),
            'sources':dict(Counter(r.get('source','unspecified') for r in rows)),
            'mask_levels':dict(Counter(r.get('mask_level','none') for r in rows)),
            'errors':errors,'passed':not errors,
            'limitations':['Exact hash check only; perceptual near-duplicate review still required.','Semantic verification cannot be inferred by this structural audit.']}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2));return not errors


def inspect_archives(root,output):
    result=[]
    for path in sorted(Path(root).rglob('*.zip')):
        try:
            with zipfile.ZipFile(path) as z:
                infos=z.infolist(); examples=[]
                for info in [i for i in infos if not i.is_dir()][:5]:
                    examples.append({'path':info.filename,'bytes':info.file_size,
                                     'prefix_hex':z.read(info)[:32].hex() if info.file_size<10_000_000 else None})
                result.append({'archive':str(path),'entries':len(infos),'uncompressed_bytes':sum(i.file_size for i in infos),'examples':examples})
        except zipfile.BadZipFile as e:
            result.append({'archive':str(path),'error':str(e)})
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('synthetic');s.add_argument('--root',default='data/synthetic')
    a=sub.add_parser('audit');a.add_argument('--manifest',required=True);a.add_argument('--root');a.add_argument('--output',default='artifacts/data_audit.json')
    b=sub.add_parser('pairs');b.add_argument('--manifest',required=True);b.add_argument('--root');b.add_argument('--split',default='train');b.add_argument('--size',type=int,default=320);b.add_argument('--output',required=True)
    i=sub.add_parser('inspect');i.add_argument('--root',default='data/downloads');i.add_argument('--output',default='artifacts/archive_inventory.json')
    args=p.parse_args()
    if args.command=='synthetic':synthetic(args.root)
    elif args.command=='audit':
        if not audit(args.manifest,args.root,args.output):raise SystemExit(1)
    elif args.command=='pairs':
        ds=ManifestDataset(args.manifest,args.split,args.size,args.root);pairs,rejected=build_pairs(ds)
        write_jsonl(args.output,pairs);print('ACCEPTED',len(pairs),'REJECTED',rejected)
    else:inspect_archives(args.root,args.output)


if __name__=='__main__':main()

