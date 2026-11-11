"""Freeze the detector; learn grasp-quality scores on independent rankfit images.

Candidate sets and geometry are unchanged. Labels are created only after feature
extraction, and only rankfit labels participate in fitting. Val selects epoch and
residual weight, including zero as an explicit unchanged-detector fallback.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from swapgrasp.data import ManifestDataset,read_jsonl,write_jsonl
from swapgrasp.geometry import decode,angle_error,rotated_iou,transform_grasps
from swapgrasp.model import SwapGrasp,TextEncoder
from swapgrasp.metrics import extended_scores


@torch.no_grad()
def extract(model,text,dataset,device,output):
    model.eval();features_all=[];bases=[];valids=[];proposals=[]
    for start in range(0,len(dataset.rows),32):
        batch=dataset.batch(range(start,min(start+32,len(dataset.rows))),with_targets=False)
        embeddings=text(batch['texts'])
        maps,features=model(batch['images'].to(device),embeddings,batch['mapping'].to(device),return_features=True)
        for i,row in enumerate(batch['rows']):
            candidates=decode(maps[i],batch['valid'][i].to(device))
            feature=features[i].float();height,width=maps.shape[-2:];values=[]
            for candidate in candidates:
                x,y,w,h,theta=candidate['grasp']
                yy,xx=torch.meshgrid(torch.linspace(-.5,.5,3,device=device),
                                     torch.linspace(-.5,.5,3,device=device),indexing='ij')
                gx=x+xx*w*math.cos(theta)-yy*h*math.sin(theta)
                gy=y+xx*w*math.sin(theta)+yy*h*math.cos(theta)
                grid=torch.stack([2*(gx+.5)/width-1,2*(gy+.5)/height-1],-1)[None]
                roi=F.grid_sample(feature[None],grid,align_corners=False,padding_mode='zeros').flatten()
                ix,iy=min(width-1,max(0,round(x))),min(height-1,max(0,round(y)))
                local=feature[:,iy,ix]
                score=np.clip(candidate['score'],1e-6,1-1e-6)
                geometry=torch.tensor([x/width,y/height,math.log(w/width),math.log(h/height),
                                        math.cos(2*theta),math.sin(2*theta),math.log(score/(1-score))],device=device)
                values.append(torch.cat([roi,local,embeddings[i].float(),geometry]))
                candidate['grasp']=transform_grasps([candidate['grasp']],batch['transforms'][i],inverse=True)[0].tolist()
            n=len(values);dimension=features.shape[1]*10+embeddings.shape[1]+7
            packed=torch.stack(values) if values else torch.zeros((0,dimension),device=device)
            features_all.append(F.pad(packed,(0,0,0,5-n)).cpu())
            bases.append([math.log(np.clip(p['score'],1e-6,1-1e-6)/(1-np.clip(p['score'],1e-6,1-1e-6))) for p in candidates]+[0.]*(5-n))
            valids.append([True]*n+[False]*(5-n))
            proposals.append({'sample_id':row['sample_id'],'image_id':row['image_id'],'predictions':candidates})
        if start%320==0:print('EXTRACT',dataset.rows[0]['split'],start,'/',len(dataset.rows),flush=True)
    cache={'features':torch.stack(features_all),'base_logits':torch.tensor(bases,dtype=torch.float32),
           'valid':torch.tensor(valids),'proposals':proposals}
    torch.save(cache,output)
    return cache


def labels(dataset,cache):
    # No GT enters extract(), candidate coordinates, or candidate features.
    ys=[];quality=[]
    for row,proposal in zip(dataset.rows,cache['proposals']):
        standard=[];strict=[];ious=[]
        assert row['sample_id']==proposal['sample_id']
        for candidate in proposal['predictions']:
            overlap=[rotated_iou(candidate['grasp'],g) for g in row['grasps']]
            angle=[math.degrees(angle_error(candidate['grasp'][4],g[4])) for g in row['grasps']]
            standard.append(any(i>.25 and a<30 for i,a in zip(overlap,angle)))
            strict.append(any(i>.50 and a<15 for i,a in zip(overlap,angle)))
            ious.append(max([i for i,a in zip(overlap,angle) if a<30] or [0.]))
        n=len(standard)
        ys.append(list(zip(standard+[False]*(5-n),strict+[False]*(5-n))))
        quality.append(ious+[0.]*(5-n))
    return torch.tensor(ys,dtype=torch.float32),torch.tensor(quality)


class Ranker(nn.Module):
    def __init__(self,dimension):
        super().__init__();self.net=nn.Sequential(nn.Linear(dimension,128),nn.ReLU(),
           nn.Dropout(.2),nn.Linear(128,32),nn.ReLU(),nn.Linear(32,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(x)


def reordered(cache,residual,alpha):
    values=cache['base_logits']+alpha*(residual[:,:,0]+.25*residual[:,:,1])
    values=values.masked_fill(~cache['valid'],-1e9)
    orders=values.argsort(dim=1,descending=True,stable=True)
    predictions=[]
    for row,order in zip(cache['proposals'],orders.tolist()):
        n=len(row['predictions']);candidates=[row['predictions'][i] for i in order if i<n]
        predictions.append({**row,'predictions':candidates})
    return predictions


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--pairs',required=True);p.add_argument('--stage',choices=['fit','test'],default='fit')
    p.add_argument('--epochs',type=int,default=40);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);device=torch.device('cuda')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);config=ck['config'];seed=config['seed']
    torch.set_num_threads(4);torch.manual_seed(seed+100);np.random.seed(seed+100);random.seed(seed+100)
    mc=config['model'];model=SwapGrasp(mc.get('channels',128),pretrained=False).to(device);model.load_state_dict(ck['model'])
    model.requires_grad_(False)
    text=TextEncoder(device,mc.get('text_mode','clip'),pretrained=mc.get('text_pretrained','openai'))
    datasets={split:ManifestDataset(a.manifest,split,config['data'].get('image_size',320)) for split in (
              ('rankfit','val') if a.stage=='fit' else ('test',))}
    caches={}
    for split,ds in datasets.items():
        path=out/(split+'_cache.pt')
        caches[split]=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else extract(model,text,ds,device,path)
    del model,text;torch.cuda.empty_cache()
    pairs=read_jsonl(a.pairs)
    if a.stage=='fit':
        fit=caches['rankfit'];val=caches['val'];y,q=labels(datasets['rankfit'],fit);yv,_=labels(datasets['val'],val)
        flat=fit['features'][fit['valid']];mean=flat.mean(0);std=flat.std(0).clamp_min(.05)
        x=((fit['features']-mean)/std).clamp(-10,10).to(device);xv=((val['features']-mean)/std).clamp(-10,10).to(device)
        ys=y.to(device);qs=q.to(device);valid=fit['valid'].to(device)
        ranker=Ranker(x.shape[-1]).to(device);optimizer=torch.optim.AdamW(ranker.parameters(),lr=3e-4,weight_decay=.01)
        writer=SummaryWriter(str(out/'tensorboard'));best=(-1.,-1.);history=[]
        for epoch in range(a.epochs+1):
            if epoch:
                ranker.train();total=0.;steps=0
                for indices in torch.randperm(len(x),device=device).split(128):
                    prediction=ranker(x[indices]);target=ys[indices];mask=valid[indices]
                    point=F.binary_cross_entropy_with_logits(prediction,target,reduction='none').mean(-1)
                    positive=(target[:,:,0]>.5)&mask;negative=~positive&mask
                    loss=.5*(point[positive].mean() if positive.any() else point.sum()*0)+.5*(point[negative].mean() if negative.any() else point.sum()*0)
                    rankable=positive.any(1)
                    if rankable.any():
                        weights=(qs[indices]*positive).masked_fill(~mask,0)
                        weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
                        logits=prediction[:,:,0].masked_fill(~mask,-1e4)
                        loss=loss+.25*(-weights*logits.log_softmax(1)).sum(1)[rankable].mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(ranker.parameters(),1.);optimizer.step()
                    total+=float(loss.detach());steps+=1
                writer.add_scalar('train/rank_loss',total/max(steps,1),epoch)
            ranker.eval()
            with torch.no_grad():residual=ranker(xv).cpu()
            best_epoch=None
            for alpha in (0.,.25,.5,1.,2.):
                scores=(val['base_logits']+alpha*(residual[:,:,0]+.25*residual[:,:,1])).masked_fill(~val['valid'],-1e9)
                idx=scores.argmax(1);success=yv[torch.arange(len(idx)),idx]
                value=(float(success[:,0].mean()),float(success[:,1].mean()))
                if value>best:
                    best=value;best_epoch={'epoch':epoch,'alpha':alpha,'val_top1':value[0],'val_strict_top1':value[1]}
                    torch.save({'model':ranker.state_dict(),'mean':mean,'std':std,'choice':best_epoch,
                        'detector_checkpoint':a.checkpoint,'detector_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()},out/'best.pt')
            record={'epoch':epoch,'best_val_top1':best[0],'best_val_strict':best[1]}
            history.append(record);writer.add_scalar('val/best_top1',best[0],epoch)
        writer.close();chosen=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        (out/'choice.json').write_text(json.dumps(chosen['choice'],indent=2));(out/'history.json').write_text(json.dumps(history,indent=2))
        ranker.load_state_dict(chosen['model']);ranker.eval()
        xv=((val['features']-chosen['mean'])/chosen['std']).clamp(-10,10).to(device)
        with torch.no_grad():residual=ranker(xv).cpu()
        predictions=reordered(val,residual,chosen['choice']['alpha']);write_jsonl(out/'val_predictions.jsonl',predictions)
        report,per_query=extended_scores(datasets['val'].rows,predictions,pairs)
        (out/'val_metrics.json').write_text(json.dumps(report,indent=2));write_jsonl(out/'val_per_query.jsonl',per_query)
        print('FITTED',json.dumps(chosen['choice']),flush=True)
    else:
        chosen=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        assert chosen['detector_sha256']==hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
        cache=caches['test']
        baseline=read_jsonl(Path(a.checkpoint).parent/'test_predictions.jsonl')
        for extracted,canonical in zip(cache['proposals'],baseline):
            assert extracted['sample_id']==canonical['sample_id']
            assert len(extracted['predictions'])==len(canonical['predictions'])
            for old,new in zip(extracted['predictions'],canonical['predictions']):
                assert np.allclose(old['grasp'],new['grasp'],atol=1e-5,rtol=1e-6)
                assert abs(old['score']-new['score'])<1e-6
        cache['proposals']=baseline
        torch.save(cache,out/'test_cache.pt')
        ranker=Ranker(cache['features'].shape[-1]).to(device);ranker.load_state_dict(chosen['model']);ranker.eval()
        x=((cache['features']-chosen['mean'])/chosen['std']).clamp(-10,10).to(device)
        with torch.no_grad():residual=ranker(x).cpu()
        predictions=reordered(cache,residual,chosen['choice']['alpha']);write_jsonl(out/'test_predictions.jsonl',predictions)
        # Sorting must preserve every exact candidate rectangle.
        for old,new in zip(cache['proposals'],predictions):
            assert sorted(tuple(p['grasp']) for p in old['predictions'])==sorted(tuple(p['grasp']) for p in new['predictions'])
        report,per_query=extended_scores(datasets['test'].rows,predictions,pairs)
        (out/'test_metrics.json').write_text(json.dumps(report,indent=2));write_jsonl(out/'test_per_query.jsonl',per_query)
        print('TESTED',json.dumps(report['standard_metrics']),flush=True)


if __name__=='__main__':main()
