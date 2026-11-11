"""Fixed-scale heatmaps, 2x2 prompt-swap panels and failure summaries."""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import textwrap
import numpy as np
from PIL import Image,ImageDraw
import torch
from swapgrasp.data import ManifestDataset,read_jsonl,letterbox
from swapgrasp.geometry import decode,transform_grasps,grasp_success
from swapgrasp.model import SwapGrasp,TextEncoder
from swapgrasp.visualization import overlay


def load(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False);mc=ck['config']['model']
    net=SwapGrasp(mc.get('channels',128),pretrained=False).to(device)
    net.load_state_dict(ck['model']);net.eval()
    text=TextEncoder(device,mc.get('text_mode','clip'),pretrained=mc.get('text_pretrained','openai'))
    return net,text,ck['config']


@torch.no_grad()
def panel(net,text,dataset,ids,device,label):
    batch=dataset.batch(ids,with_targets=False)
    if text.mode=='debug_hash' and any(r.get('source')!='synthetic_engineering_fixture' for r in batch['rows']):
        raise ValueError('Synthetic debug model cannot explain real data')
    output=net(batch['images'].to(device),text(batch['texts']),batch['mapping'].to(device))
    tiles=[]
    for i,row in enumerate(batch['rows']):
        # Labels accessed only after the model has produced all outputs.
        with Image.open(dataset.path(row['image'])) as im:
            image,_,_=letterbox(im.convert('RGB'),dataset.size)
        ps=decode(output[i],batch['valid'][i].to(device))
        gt=transform_grasps(row['grasps'],batch['transforms'][i])
        image=overlay(image,gt,ps,quality=output[i,0].sigmoid().cpu().numpy())
        tile=Image.new('RGB',(dataset.size,dataset.size+64),(245,246,249));tile.paste(image,(0,64))
        caption=label+' | '+row['text']
        ImageDraw.Draw(tile).multiline_text((6,5),'\n'.join(textwrap.wrap(caption,width=max(20,dataset.size//7)))[:180],fill=(20,30,45))
        tiles.append(tile)
    return tiles


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--baseline-checkpoint')
    p.add_argument('--manifest',required=True);p.add_argument('--root');p.add_argument('--pairs',required=True)
    p.add_argument('--split',default='test');p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=8)
    p.add_argument('--predictions');p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);device=torch.device(a.device)
    net,text,config=load(a.checkpoint,device);dataset=ManifestDataset(a.manifest,a.split,config['data']['image_size'],a.root)
    baseline=None
    if a.baseline_checkpoint:
        baseline=load(a.baseline_checkpoint,device)
        if baseline[2]['data']['image_size']!=dataset.size:raise ValueError('Comparison needs same image size')
    candidates=[p for p in read_jsonl(a.pairs) if p['sample_a'] in dataset.by_id and p['sample_b'] in dataset.by_id]
    pairs=[];seen_images=set()
    for pair in candidates:
        if pair['image_id'] in seen_images:continue
        pairs.append(pair);seen_images.add(pair['image_id'])
        if len(pairs)>=a.limit:break
    for pair in pairs:
        ids=[dataset.by_id[pair['sample_a']],dataset.by_id[pair['sample_b']]]
        method_tiles=panel(net,text,dataset,ids,device,'Method')
        columns=[method_tiles]
        if baseline:columns.insert(0,panel(baseline[0],baseline[1],dataset,ids,device,'Baseline'))
        w,h=method_tiles[0].size;combined=Image.new('RGB',(w*len(columns),h*2))
        for col,tiles in enumerate(columns):
            for row,tile in enumerate(tiles):combined.paste(tile,(col*w,row*h))
        combined.save(out/(pair['pair_id']+'.png'))
    if a.predictions:
        predictions={p['sample_id']:p['predictions'] for p in read_jsonl(a.predictions)}
        by_image=defaultdict(list)
        for row in dataset.rows:by_image[row['image_id']].append(row)
        failures=[]
        for row in dataset.rows:
            ps=predictions[row['sample_id']]
            if grasp_success(ps,row['grasps']):kind='success'
            elif not ps:kind='no_valid_prediction'
            elif any(grasp_success(ps,other['grasps']) for other in by_image[row['image_id']] if other['object_id']!=row['object_id']):
                kind='matches_other_annotated_object'
            else:kind='unresolved_localization_or_geometry'
            failures.append({'sample_id':row['sample_id'],'image_id':row['image_id'],'failure_category':kind})
        with open(out/'failure_cases.csv','w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(failures[0]));writer.writeheader();writer.writerows(failures)
        counts=Counter(r['failure_category'] for r in failures)
        (out/'failure_counts.json').write_text(json.dumps(counts,indent=2))
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,4));ax.barh(list(counts),list(counts.values()))
        ax.set_xlabel('Query count');fig.tight_layout();fig.savefig(out/'failure_counts.png',dpi=160);plt.close(fig)
    (out/'README.txt').write_text('Heatmap scale is fixed to sigmoid quality [0,1] in every panel. Green: all GT. Red: top-1. Amber: other candidates. Rows change instructions; columns compare models when baseline is supplied. Failure tags are annotation-based diagnostics, not physical execution outcomes.\n')
    print('Explanation panels:',len(pairs),out)


if __name__=='__main__':main()
