"""Separate prediction and scoring: inference reads only RGB, text, and IDs."""
import argparse
import json
import os
from pathlib import Path
import torch
from swapgrasp.data import ManifestDataset,read_jsonl,write_jsonl
from swapgrasp.model import SwapGrasp,TextEncoder
from swapgrasp.metrics import extended_scores,paired_bootstrap
from train import predict


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    ip=sub.add_parser('predict'); ip.add_argument('--checkpoint',required=True)
    ip.add_argument('--manifest',required=True); ip.add_argument('--split',default='test')
    ip.add_argument('--root'); ip.add_argument('--output',required=True)
    ip.add_argument('--visualize-dir',help='Save prediction PNGs, heatmaps and index.html without reading GT')
    ip.add_argument('--batch-size',type=int,help='Number of instructions per inference batch')
    ip.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    sp=sub.add_parser('score'); sp.add_argument('--manifest',required=True); sp.add_argument('--split',default='test')
    sp.add_argument('--predictions',required=True); sp.add_argument('--pairs'); sp.add_argument('--output',required=True)
    cp=sub.add_parser('compare'); cp.add_argument('--scores-a',required=True); cp.add_argument('--scores-b',required=True)
    cp.add_argument('--output',required=True)
    a=p.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
    if a.command=='predict':
        device=torch.device(a.device); ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
        config=ck['config']; mc=config['model']; size=config['data'].get('image_size',320)
        dataset=ManifestDataset(a.manifest,a.split,size,a.root)
        if mc.get('text_mode','clip')=='debug_hash' and any(r.get('source')!='synthetic_engineering_fixture' for r in dataset.rows):
            raise ValueError('Synthetic debug checkpoint cannot evaluate real data')
        model=SwapGrasp(mc.get('channels',128),pretrained=False).to(device); model.load_state_dict(ck['model'])
        text=TextEncoder(device,mc.get('text_mode','clip'),pretrained=mc.get('text_pretrained','openai'))
        batch=a.batch_size if a.batch_size is not None else min(64,config['sampler']['queries'])
        if batch<1:raise ValueError('Batch size must be positive')
        ps=predict(model,text,dataset,device,batch,a.visualize_dir); write_jsonl(a.output,ps)
        print('Predictions saved:',len(ps),a.output)
    elif a.command=='score':
        rows=[r for r in read_jsonl(a.manifest) if r['split']==a.split]
        if not rows: raise ValueError('Empty evaluation split')
        predictions=read_jsonl(a.predictions);pairs=read_jsonl(a.pairs) if a.pairs else []
        detailed,detailed_scores=extended_scores(rows,predictions,pairs)
        metrics=detailed['standard_metrics']
        fields=['sample_id','image_id','category_split','grasp_at1','grasp_at5']
        scores=[{key:r[key] for key in fields} for r in detailed_scores]
        out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2))
        (out/'detailed_metrics.json').write_text(json.dumps(detailed,indent=2))
        write_jsonl(out/'per_query.jsonl',scores)
        write_jsonl(out/'detailed_per_query.jsonl',detailed_scores)
        print(json.dumps({'standard':metrics,'detailed':detailed},indent=2))
    else:
        metrics=paired_bootstrap(read_jsonl(a.scores_a),read_jsonl(a.scores_b))
        Path(a.output).write_text(json.dumps(metrics,indent=2)); print(metrics)


if __name__=='__main__': main()

