"""Predict one image and text with the fixed detector and frozen candidate ranker."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from PIL import Image
from swapgrasp.data import ManifestDataset,write_jsonl
from swapgrasp.model import SwapGrasp,TextEncoder
from swapgrasp.visualization import overlay
from scripts.rerank_candidates import Ranker,extract,reordered


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--detector',required=True);p.add_argument('--ranker',required=True)
    p.add_argument('--image',required=True);p.add_argument('--text',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.detector,map_location='cpu',weights_only=False)
    rank=torch.load(a.ranker,map_location='cpu',weights_only=False)
    with Path(a.detector).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=rank['detector_sha256']:
            raise ValueError('This ranker was fitted to a different detector checkpoint')
    config=ck['config'];device=torch.device(a.device);torch.set_num_threads(4)
    model=SwapGrasp(config['model'].get('channels',128),pretrained=False).to(device)
    model.load_state_dict(ck['model']);model.requires_grad_(False)
    text=TextEncoder(device,config['model'].get('text_mode','clip'),
                     pretrained=config['model'].get('text_pretrained','openai'))
    request={'sample_id':'user_input','image_id':'user_input','image':str(Path(a.image).resolve()),
             'text':a.text,'split':'input'}
    manifest=out/'request.jsonl';write_jsonl(manifest,[request])
    dataset=ManifestDataset(manifest,'input',config['data'].get('image_size',320))
    started=time.monotonic();cache=extract(model,text,dataset,device,out/'features.pt')
    scorer=Ranker(cache['features'].shape[-1]).to(device)
    scorer.load_state_dict(rank['model']);scorer.eval()
    features=((cache['features']-rank['mean'])/rank['std']).clamp(-10,10).to(device)
    with torch.no_grad():residual=scorer(features).cpu()
    predictions=reordered(cache,residual,rank['choice']['alpha'])
    result={'request':request,'detector':a.detector,'ranker':a.ranker,'ranker_choice':rank['choice'],
            'predictions':predictions[0]['predictions'],'original':cache['proposals'][0]['predictions'],
            'seconds':time.monotonic()-started,'units':'original image pixels; clockwise radians, pi-periodic',
            'scores':'Original detector centre scores; order reflects the learned residual ranker.'}
    (out/'prediction.json').write_text(json.dumps(result,indent=2))
    with Image.open(a.image) as image:
        overlay(image,[],result['original']).save(out/'original.png')
        overlay(image,[],result['predictions']).save(out/'reranked.png')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Reranked grasp</title><style>body{max-width:1000px;margin:30px auto;font:18px system-ui}img{width:49%}</style><p>'+
           __import__('html').escape(a.text)+'</p><p>Left: original ranking. Right: reranked. Red: first grasp; amber: remaining candidates.</p><img src="original.png"><img src="reranked.png">')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
