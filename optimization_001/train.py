"""Single-GPU reproducible baseline / SwapGrasp trainer."""
import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import platform
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from swapgrasp.data import ManifestDataset,PairSampler,read_jsonl
from swapgrasp.geometry import decode,transform_grasps
from swapgrasp.losses import grasp_loss,swap_loss,hard_swap_loss,swap_weight
from swapgrasp.metrics import score_predictions
from swapgrasp.model import SwapGrasp,TextEncoder


def F_cosine(a,b):
    return (a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-12)


@torch.no_grad()
def predict(model,text_encoder,dataset,device,batch_size=8,visualization_dir=None):
    model.eval(); results=[]
    exporter=None
    if visualization_dir:
        from swapgrasp.visualization import PredictionExporter
        exporter=PredictionExporter(visualization_dir)
    for start in range(0,len(dataset.rows),batch_size):
        batch=dataset.batch(range(start,min(start+batch_size,len(dataset.rows))),with_targets=False)
        output=model(batch['images'].to(device),text_encoder(batch['texts']),batch['mapping'].to(device))
        for i,row in enumerate(batch['rows']):
            ps=decode(output[i],batch['valid'][i].to(device))
            for p in ps:
                p['grasp']=transform_grasps([p['grasp']],batch['transforms'][i],inverse=True)[0].tolist()
            results.append({'sample_id':row['sample_id'],'image_id':row['image_id'],'predictions':ps})
            if exporter:
                exporter.add(dataset,row,ps,output[i,0].sigmoid().float().cpu().numpy(),batch['transforms'][i])
    if exporter:exporter.finish()
    return results


def fingerprint(config):
    h=hashlib.sha256()
    for key in ['manifest','pairs']:
        name=config['data'].get(key)
        if name:
            h.update(Path(name).read_bytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    p.add_argument('--resume'); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args(); config=yaml.safe_load(Path(a.config).read_text()); out=Path(a.output)
    if (out/'last.pt').exists() and not a.resume:
        raise FileExistsError('Run already has a checkpoint; use --resume or a new output')
    out.mkdir(parents=True,exist_ok=True); device=torch.device(a.device)
    seed=config.get('seed',42); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.set_num_threads(config.get('cpu_threads',4))
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    size=config['data'].get('image_size',320)
    train=ManifestDataset(config['data']['manifest'],'train',size,config['data'].get('root'))
    val=ManifestDataset(config['data']['manifest'],'val',size,config['data'].get('root'))
    if {r['image_id'] for r in train.rows} & {r['image_id'] for r in val.rows}:
        raise ValueError('Image leakage between train and validation')
    synthetic=all(r.get('source')=='synthetic_engineering_fixture' for r in train.rows+val.rows)
    if config['model'].get('text_mode','clip')=='debug_hash' and not synthetic:
        raise ValueError('debug_hash text is permitted only for synthetic engineering fixtures')
    sampler=PairSampler(train,config['data'].get('pairs'),
                        queries=config['sampler'].get('queries',config['sampler'].get('queries_per_step',8)),
                        pairs_per_step=config['sampler'].get('pairs_per_step',0),seed=seed,
                        image_balanced=config['sampler'].get('image_balanced',False))
    if config['loss']['lambda_swap']>0 and not sampler.pairs:
        raise ValueError('Swap training requires verified pairs')
    model=SwapGrasp(channels=config['model'].get('channels',128),
                    pretrained=config['model'].get('image_pretrained',True),
                    image_weights_path=config['model'].get('image_weights_path')).to(device)
    text=TextEncoder(device,mode=config['model'].get('text_mode','clip'),
                     pretrained=config['model'].get('text_pretrained','openai'))
    tc=config['train'];train.cache_images=tc.get('cache_images',0);train.cache_targets=tc.get('cache_targets',0)
    val.cache_images=tc.get('cache_images',0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=tc['lr'],weight_decay=tc.get('weight_decay',1e-4))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',factor=.5,patience=3)
    amp=tc.get('amp',True) and device.type=='cuda'
    amp_name=tc.get('amp_dtype','bfloat16' if amp and torch.cuda.is_bf16_supported() else 'float16')
    if amp_name not in ['bfloat16','float16']:
        raise ValueError('amp_dtype must be bfloat16 or float16')
    amp_dtype=getattr(torch,amp_name)
    if amp and amp_dtype==torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError('bfloat16 AMP is unsupported on this GPU')
    scaler=torch.amp.GradScaler('cuda',enabled=amp and amp_dtype==torch.float16)
    step,best=0,-1.; data_hash=fingerprint(config)
    if a.resume:
        ck=torch.load(a.resume,map_location=device,weights_only=False)
        if ck['config']!=config or ck['data_hash']!=data_hash:
            raise ValueError('Resume config/data changed; start a distinct experiment')
        model.load_state_dict(ck['model']); optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        sampler.load_state_dict(ck['sampler']); step,best=ck['step'],ck['best']
        random.setstate(ck['rng_python']); np.random.set_state(ck['rng_numpy'])
        torch.set_rng_state(ck['rng_torch'].cpu())
        if torch.cuda.is_available() and ck['rng_cuda']:
            torch.cuda.set_rng_state_all([s.cpu() for s in ck['rng_cuda']])
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    commit=subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
    env={'torch':torch.__version__,'cuda':torch.version.cuda,'device':str(device),'platform':platform.platform(),
         'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
         'data_hash':data_hash,'git_commit':commit,'synthetic_only':synthetic,
         'training_queries':len(train.rows),'training_images':len({r['image_id'] for r in train.rows}),
         'pairs':len(sampler.pairs),'seed':seed,'sampler_queries':config['sampler']['queries'],
         'pairs_per_step':config['sampler'].get('pairs_per_step',0),
         'physical_gpu':os.environ.get('SWAPGRASP_PHYSICAL_GPU')}
    env['amp_dtype']=amp_name if amp else None
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    writer=SummaryWriter(str(out/'tensorboard'),purge_step=step+1 if a.resume else None)
    log=open(out/'metrics.jsonl','a',buffering=1); schedule=open(out/'schedule.jsonl','a',buffering=1)
    pairs=read_jsonl(config['data']['pairs']) if config['data'].get('pairs') else []
    started=time.monotonic(); last_step=step
    prefetch=ThreadPoolExecutor(max_workers=1) if tc.get('prefetch',False) else None
    pending=None
    if prefetch:
        next_ids,next_pairs=sampler.next()
        pending=prefetch.submit(train.batch,next_ids)
        next_sampler_state=sampler.state_dict()
    interval_start=time.monotonic();interval_step=step
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)

    def save(name):
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'scaler':scaler.state_dict(),'sampler':sampler.state_dict(),'step':step,'best':best,
               'config':config,'data_hash':data_hash,'rng_python':random.getstate(),
               'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
               'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        temp=out/(name+'.tmp'); torch.save(state,temp); temp.replace(out/name)

    try:
        while step<tc['max_steps']:
            model.train()
            if prefetch:
                ids,pair_positions=next_ids,next_pairs
                batch=pending.result()
                consumed_sampler_state=next_sampler_state
                if step+1<tc['max_steps']:
                    next_ids,next_pairs=sampler.next()
                    pending=prefetch.submit(train.batch,next_ids)
                    next_sampler_state=sampler.state_dict()
            else:
                ids,pair_positions=sampler.next();batch=train.batch(ids)
            output=model(batch['images'].to(device),text(batch['texts']),batch['mapping'].to(device)) if not amp else None
            if amp:
                with torch.autocast('cuda',dtype=amp_dtype):
                    output=model(batch['images'].to(device),text(batch['texts']),batch['mapping'].to(device))
            base,parts=grasp_loss(output,batch['targets'],config['loss'].get('lambda_geometry',1.))
            supports=[t['support'].to(device) for t in batch['targets']]
            lc=config['loss']; extra={}
            if lc.get('swap_type')=='hard':
                swap,extra=hard_swap_loss(output[:,0],supports,pair_positions,
                     lc.get('margin',.5),lc.get('temperature',.25),lc.get('top_fraction',.25))
            else:
                swap=swap_loss(output[:,0],supports,pair_positions,
                               lc.get('margin',.2),lc.get('temperature',.25))
            weight=swap_weight(step+1,lc)
            loss=base+weight*swap
            diagnostics={}
            if (step+1)%250==0 and pair_positions:
                gb=torch.autograd.grad(base,output,retain_graph=True)[0][:,0].float().flatten()
                gs=torch.autograd.grad(swap,output,retain_graph=True)[0][:,0].float().flatten()
                diagnostics={'output_quality_gradient_cosine':float(F_cosine(gb,gs)),
                     'weighted_swap_to_base_output_gradient_ratio':float(weight*gs.norm()/gb.norm().clamp_min(1e-12))}
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step+1}')
            optimizer.zero_grad(set_to_none=True); scaler.scale(loss).backward()
            scaler.unscale_(optimizer); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            warmup=tc.get('warmup_steps',500)
            if step<warmup:
                for group in optimizer.param_groups:
                    group['lr']=tc['lr']*(step+1)/max(warmup,1)
            scaler.step(optimizer); scaler.update(); step+=1
            schedule.write(json.dumps({'step':step,'sample_ids':[train.rows[i]['sample_id'] for i in ids],
                                       'pair_positions':pair_positions})+'\n')
            if step%tc.get('log_every',20)==0 or step==1:
                record={'step':step,'loss':float(loss.detach()),'quality':float(parts['quality'].detach()),
                        'geometry':float(parts['geometry'].detach()),'swap':float(swap.detach()),
                        'gradient_norm':float(norm),'lr':optimizer.param_groups[0]['lr'],
                        'weighted_swap':float((weight*swap).detach()),
                        'batch_queries':len(ids),'samples_seen':step*config['sampler']['queries'],
                        'base_loss':float(base.detach()),'lambda_swap':weight,
                        'queries_per_second':(step-interval_step)*len(ids)/max(time.monotonic()-interval_start,1e-6),
                        'gpu_peak_allocated_gb':torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.,
                        'gpu_peak_reserved_gb':torch.cuda.max_memory_reserved(device)/2**30 if device.type=='cuda' else 0.,
                        'seconds':time.monotonic()-started,
                        **{k:float(v) for k,v in extra.items()},**diagnostics}
                log.write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
                for key,value in record.items():
                    if key!='step': writer.add_scalar('train/'+key,value,step)
                writer.flush();interval_start=time.monotonic();interval_step=step
            if step%tc.get('validate_every',500)==0 or step==tc['max_steps']:
                predictions=predict(model,text,val,device,min(64,config['sampler']['queries']))
                metrics,_=score_predictions(val.rows,predictions,pairs)
                record={'step':step,**{'val/'+k:v for k,v in metrics.items()}}
                log.write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
                for key,value in metrics.items():
                    if isinstance(value,(int,float)): writer.add_scalar('val/'+key,value,step)
                if step>=warmup: scheduler.step(metrics['grasp_at1'])
                if prefetch:sampler.load_state_dict(consumed_sampler_state)
                if metrics['grasp_at1']>best:
                    best=metrics['grasp_at1']; save('best.pt')
                save('last.pt'); writer.flush()
                if prefetch:sampler.load_state_dict(next_sampler_state)
            if step%tc.get('snapshot_every',2000)==0:
                if prefetch:sampler.load_state_dict(consumed_sampler_state)
                save(f'step_{step}.pt')
                if prefetch:sampler.load_state_dict(next_sampler_state)
        print('FINISHED',step,'steps;',round(time.monotonic()-started,1),'seconds',flush=True)
    finally:
        if prefetch:prefetch.shutdown(wait=True)
        writer.close(); log.close(); schedule.close()


if __name__=='__main__':
    main()

