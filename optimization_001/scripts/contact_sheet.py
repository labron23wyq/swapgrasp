"""Render deterministic real-data inspection sheets with masks and GT grasps."""
import argparse
from pathlib import Path
import random
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import numpy as np
from PIL import Image,ImageDraw
from swapgrasp.data import ManifestDataset,load_target_mask
from swapgrasp.visualization import overlay

p=argparse.ArgumentParser()
p.add_argument('--manifest',required=True);p.add_argument('--split',default='train')
p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=24)
a=p.parse_args();ds=ManifestDataset(a.manifest,a.split)
rows=ds.rows.copy();random.Random(42).shuffle(rows);rows=rows[:a.limit]
tiles=[]
for row in rows:
    with Image.open(ds.path(row['image'])) as source:image=source.convert('RGB')
    mask=load_target_mask(row,ds.path(row['mask']))
    tinted=np.asarray(image).copy()
    tinted[mask]=(tinted[mask]*.65+np.array([40,120,255])*.35).astype(np.uint8)
    image=overlay(Image.fromarray(tinted),row['grasps'])
    image.thumbnail((320,320));tile=Image.new('RGB',(340,396),'white');tile.paste(image,(10,70))
    import textwrap
    caption=row['text']+f'\n{len(row["grasps"])} GT | object {row["object_id"]}, part {row["part_id"]}'
    caption+='\n'+row['sample_id'][:12]
    ImageDraw.Draw(tile).multiline_text((8,5),'\n'.join(textwrap.wrap(caption,43)),fill=(15,25,40))
    tiles.append(tile)
columns=4;sheet=Image.new('RGB',(columns*340,((len(tiles)+columns-1)//columns)*396),(225,229,235))
for i,tile in enumerate(tiles):sheet.paste(tile,((i%columns)*340,(i//columns)*396))
Path(a.output).parent.mkdir(parents=True,exist_ok=True);sheet.save(a.output)
print('CONTACT SHEET',a.output,len(tiles),flush=True)
