"""Inspect accepted pairs: two instructions and their distinct GT/mask overlays."""
import argparse
from pathlib import Path
import random,sys,textwrap
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import numpy as np
from PIL import Image,ImageDraw
from swapgrasp.data import ManifestDataset,read_jsonl,load_target_mask
from swapgrasp.visualization import overlay

p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--pairs',required=True)
p.add_argument('--split',default='train');p.add_argument('--limit',type=int,default=16);p.add_argument('--output',required=True)
a=p.parse_args();ds=ManifestDataset(a.manifest,a.split);pairs=[r for r in read_jsonl(a.pairs) if r['split']==a.split]
random.Random(42).shuffle(pairs);chosen=[];seen=set()
for pair in pairs:
    if pair['image_id'] in seen:continue
    chosen.append(pair);seen.add(pair['image_id'])
    if len(chosen)>=a.limit:break
tiles=[]
for pair in chosen:
    tile=Image.new('RGB',(672,390),'white')
    for j,key in enumerate(['sample_a','sample_b']):
        row=ds.rows[ds.by_id[pair[key]]]
        with Image.open(ds.path(row['image'])) as source:image=source.convert('RGB')
        mask=load_target_mask(row,ds.path(row['mask']));tinted=np.asarray(image).copy()
        tinted[mask]=(tinted[mask]*.65+np.array([40,120,255])*.35).astype(np.uint8)
        image=overlay(Image.fromarray(tinted),row['grasps']);image.thumbnail((320,320))
        tile.paste(image,(j*336+8,65))
        text=row['text']+'\n'+pair['image_id'][:12]+f' | object {row["object_id"]}'
        ImageDraw.Draw(tile).multiline_text((j*336+8,5),'\n'.join(textwrap.wrap(text,43)),fill=(15,25,40))
    tiles.append(tile)
if not tiles:raise ValueError('No accepted pairs')
sheet=Image.new('RGB',(1344,((len(tiles)+1)//2)*390),(220,225,233))
for i,tile in enumerate(tiles):sheet.paste(tile,((i%2)*672,(i//2)*390))
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);sheet.save(path)
path.with_suffix('.json').write_text(__import__('json').dumps(chosen,indent=2))
print('PAIR SHEET',path,'images',len(chosen),flush=True)
