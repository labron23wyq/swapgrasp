"""Strict adapter for inspected Grasp-Anything++ archives.

Labels must be complete and SHA-verified. Images can be selectively fetched
from the official split ZIP over HTTP ranges, with per-member CRC provenance.
Creates a separately named image-disjoint diagnostic protocol.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import random
import zipfile
import numpy as np
import torch
from PIL import Image
from swapgrasp.data import write_jsonl,ManifestDataset,build_pairs
from swapgrasp.remote_zip import RemoteConcat,segments_from_catalog
from swapgrasp.prefix_zip import PrefixZip


def verified_zip(path,allow_prefix=False):
    path=Path(path)
    receipt=path.with_name(path.name+'.verified.json')
    if not path.exists() or not receipt.exists():
        if allow_prefix:
            candidate=path if path.exists() else path.with_name(path.name+'.part')
            archive=PrefixZip(candidate)
            print('PREFIX CRC ONLY',path,len(archive.entries),'complete members',flush=True)
            return archive
        raise FileNotFoundError(f'Wait for complete SHA-verified archive: {path}')
    metadata=json.loads(receipt.read_text())
    if path.stat().st_size!=metadata['size']:
        raise ValueError('Archive size differs from verification receipt')
    archive=zipfile.ZipFile(path)
    archive.integrity='full_archive_lfs_sha256_and_member_crc'
    return archive


def target_key(name):
    stem=Path(name).stem
    image_id,obj,part=stem.rsplit('_',2)
    if len(image_id)!=64 or not all(c in '0123456789abcdef' for c in image_id):
        raise ValueError(f'Unexpected image ID {image_id}')
    return image_id,int(obj),int(part)


def canonical_grasps(values):
    if not isinstance(values,torch.Tensor) or values.ndim!=2 or values.shape[1]!=6:
        raise ValueError('Expected [N,6] grasp Tensor: score,x,y,w,h,source_degrees')
    raw=values.detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(raw).all() or np.any(raw[:,3:5]<=0):
        raise ValueError('Invalid grasp values')
    result=raw[:,1:].copy()
    # Official Grasp has counterclockwise angle and explicitly negates source.
    # Our clockwise y-down angle therefore equals source_degrees in radians.
    result[:,4]=(np.deg2rad(result[:,4])+np.pi/2)%np.pi-np.pi/2
    return result.astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--downloads',default='data/downloads')
    p.add_argument('--output',default='data/processed');p.add_argument('--reference',default='third_party/LGD')
    p.add_argument('--train-images',type=int,default=1000);p.add_argument('--val-images',type=int,default=200)
    p.add_argument('--test-images',type=int,default=200);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--minimum-objects',type=int,default=1,help='Diagnostic selection: require this many officially allowed object IDs per image')
    p.add_argument('--local-image-zip');p.add_argument('--endpoint',default='https://hf-mirror.com')
    p.add_argument('--verified-local-images',action='store_true',help='Require verified split image files; disable HTTP fallback')
    p.add_argument('--quarantine-subpixel',action='store_true',help='Exclude queries with a box whose xywh all lie in [0, 1]; preserve the source record in quarantine.jsonl')
    p.add_argument('--allow-prefix-annotations',action='store_true',help='Diagnostic only: CRC-checked complete members in current download prefixes')
    a=p.parse_args();down=Path(a.downloads);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    splitroot=Path(a.reference)/'split/grasp-anything++'
    official={}
    for name,relative in [('train','train/seen.obj'),('seen','test/seen.obj'),('unseen','test/unseen.obj')]:
        with open(splitroot/relative,'rb') as f:values=pickle.load(f)
        if not isinstance(values,list) or not all(isinstance(v,str) for v in values):raise ValueError('Unexpected official split list')
        official[name]=set(values)
    test_images={v.rsplit('_',1)[0] for group in ['seen','unseen'] for v in official[group]}
    if a.minimum_objects<1:raise ValueError('minimum-objects must be positive')
    train_counts=Counter(v.rsplit('_',1)[0] for v in official['train'])
    test_counts=Counter(v.rsplit('_',1)[0] for v in (official['seen']|official['unseen']))
    train_images={iid for iid,count in train_counts.items() if count>=a.minimum_objects}-test_images
    eligible_test_images={iid for iid,count in test_counts.items() if count>=a.minimum_objects}
    rng=random.Random(a.seed);train_images=sorted(train_images);rng.shuffle(train_images)
    requested=a.train_images+a.val_images
    selected=train_images[:requested]
    train_ids=set(selected[:a.train_images]);val_ids=set(selected[a.train_images:])
    tests=sorted(eligible_test_images);rng.shuffle(tests);test_ids=set(tests[:a.test_images])
    wanted=train_ids|val_ids|test_ids
    protocol_name='image_disjoint_diagnostic_prefix_subset' if a.allow_prefix_annotations else 'image_disjoint_diagnostic_subset'
    protocol={'name':protocol_name,'seed':a.seed,
              'official_train_test_overlapping_image_ids':sorted({v.rsplit('_',1)[0] for v in official['train']}&test_images),
              'train_images_requested':a.train_images,'val_images_requested':a.val_images,
              'test_images_requested':a.test_images,'selected_train_image_ids':sorted(train_ids),
              'minimum_allowed_objects_per_image':a.minimum_objects,
              'eligible_train_images':len(train_images),'eligible_test_images':len(eligible_test_images),
              'selected_val_image_ids':sorted(val_ids),'selected_test_image_ids':sorted(test_ids),
              'note':'Official holdout object IDs preserved; training images sharing any holdout image removed. Validation held out by image. This is not an unchanged official protocol.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    positives=verified_zip(down/'Grasp-Anything-pp/grasp_label_positive.zip',a.allow_prefix_annotations)
    instructions=verified_zip(down/'Grasp-Anything-pp/grasp_instructions.zip',a.allow_prefix_annotations)
    masks=verified_zip(down/'Grasp-Anything-pp/part_mask.zip',a.allow_prefix_annotations)
    scenes=verified_zip(down/'Grasp-Anything/scene_description.zip',a.allow_prefix_annotations)
    archive_integrity={name:archive.integrity for name,archive in [('positive',positives),('instructions',instructions),('masks',masks),('scenes',scenes)]}
    print('Indexing labels',flush=True)
    candidates=[]
    for info in positives.infolist():
        if not info.filename.endswith('.pt'):continue
        iid,obj,part=target_key(info.filename)
        if iid not in wanted:continue
        oid=f'{iid}_{obj}'
        allowed=(oid in official['train']) if iid not in test_ids else (oid in official['seen'] or oid in official['unseen'])
        if allowed:candidates.append((info,iid,obj,part))
    print('Selected',len(candidates),'queries across',len(wanted),'requested images',flush=True)
    if a.local_image_zip and a.verified_local_images:
        p.error('--local-image-zip and --verified-local-images are mutually exclusive')
    if a.local_image_zip:
        images=zipfile.ZipFile(a.local_image_zip);remote=None
    else:
        catalog=json.loads((down/'catalog.json').read_text())
        segments=segments_from_catalog(catalog,'airvlab/Grasp-Anything',['image_part_aa','image_part_ab'])
        for segment in segments:
            path=down/'Grasp-Anything'/segment['path']
            if a.verified_local_images:
                receipt=json.loads(path.with_name(path.name+'.verified.json').read_text())
                expected=catalog[segment['repo']]['files'][segment['path']]
                if (path.stat().st_size, receipt['size'], receipt['sha256'], receipt['revision'])!=(segment['size'], segment['size'], expected['lfs']['oid'], segment['revision']):
                    raise ValueError(f'Image archive receipt mismatch: {path}')
            segment['local_path']=str(path if path.exists() else path.with_name(path.name+'.part'))
        remote=RemoteConcat(segments,a.endpoint,cache_dir=down/'range_cache',local_only=a.verified_local_images)
        print('Reading image ZIP central directory',flush=True)
        images=zipfile.ZipFile(remote)
    records=[];rejected=Counter();manifest_ids=set();quarantine=[]
    for idx,(info,iid,obj,part) in enumerate(candidates):
        stem=f'{iid}_{obj}_{part}';oid=f'{iid}_{obj}'
        try:
            instruction=pickle.loads(instructions.read(f'grasp_instructions/{stem}.pkl'))
            if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Invalid instruction')
            grasps=canonical_grasps(torch.load(io.BytesIO(positives.read(info)),map_location='cpu',weights_only=True))
            if len(grasps)==0:raise ValueError('Empty grasp labels')
            split='test' if iid in test_ids else ('val' if iid in val_ids else 'train')
            if a.quarantine_subpixel and np.any(np.all((grasps[:,:4]>=0)&(grasps[:,:4]<=1),axis=1)):
                quarantine.append({'sample_id':stem,'split':split,'reason':'all_xywh_in_unit_interval','grasps':grasps.tolist()})
                raise ValueError('Quarantined ambiguous subpixel geometry')
            rawmask=masks.read(f'part_mask/{stem}.npy')
            mask=np.load(io.BytesIO(rawmask),allow_pickle=False)
            if mask.ndim!=2 or not np.isin(mask,[0,1,255]).all() or not mask.any():raise ValueError('Invalid binary part mask')
            image_path=out/'image'/f'{iid}.jpg';image_path.parent.mkdir(exist_ok=True)
            if not image_path.exists():image_path.write_bytes(images.read(f'image/{iid}.jpg'))
            with Image.open(image_path) as im:w,h=im.size
            if mask.shape!=(h,w):raise ValueError('Mask/image size mismatch')
            if np.any(grasps[:,0]<0) or np.any(grasps[:,0]>=w) or np.any(grasps[:,1]<0) or np.any(grasps[:,1]>=h):raise ValueError('GT centre out of bounds')
            px=np.floor(grasps[:,0]+.5).astype(int);py=np.floor(grasps[:,1]+.5).astype(int)
            inside=(px>=0)&(px<w)&(py>=0)&(py<h)
            membership=inside.copy();membership[inside]=mask[py[inside],px[inside]]>0
            # Retain valid geometry for baseline; pair builder rejects any target
            # with a centre outside its provided mask. Never silently drop GT.
            scene,objects=pickle.loads(scenes.read(f'scene_description/{iid}.pkl'))
            object_name=objects[obj]
            if not isinstance(object_name,str):raise ValueError('Object lookup failed')
            mask_path=out/'part_mask'/f'{stem}.npy';mask_path.parent.mkdir(exist_ok=True)
            np.save(mask_path,mask.astype(bool,copy=False),allow_pickle=False)
            row={'sample_id':stem,'image_id':iid,'image':f'image/{iid}.jpg','text':instruction.strip(),
                 'object_id':obj,'part_id':part,'object_name':object_name,'scene_description':scene,
                 'mask':f'part_mask/{stem}.npy','mask_level':'part','grasps':grasps.tolist(),
                 'angle_unit':'radian','split':split,'source':'airvlab/Grasp-Anything-pp',
                 'category_split':'unseen' if oid in official['unseen'] else 'seen',
                 'mapping_verified':True,'semantic_verified':False,
                 'mapping_verification':'exact_official_image_object_part_filename',
                 'semantic_verification_scope':'dataset_mapping_only_not_human_confirmation',
                 'grasp_center_part_membership_fraction':float(membership.mean()),
                 'image_integrity':'full_archive_lfs_sha256_and_member_crc' if a.verified_local_images else ('zip_crc_only_full_archive_sha_pending' if remote else 'local_zip'),
                 'annotation_integrity':archive_integrity,
                 'protocol':protocol_name}
            records.append(row);manifest_ids.add(iid)
        except (KeyError,ValueError,IndexError) as e:
            rejected[str(e)]+=1
        if (idx+1)%100==0:print('Prepared',idx+1,'/',len(candidates),'accepted',len(records),flush=True)
    write_jsonl(out/'manifest.jsonl',records)
    if a.quarantine_subpixel:write_jsonl(out/'quarantine.jsonl',quarantine)
    all_pairs=[];pair_summary={}
    for split in ['train','val','test']:
        if not any(r['split']==split for r in records):continue
        ds=ManifestDataset(out/'manifest.jsonl',split,320)
        print('Building pairs',split,len(ds.rows),flush=True)
        ps,reasons=build_pairs(ds);all_pairs.extend(ps);pair_summary[split]={'accepted':len(ps),'images':len({p['image_id'] for p in ps}),'rejected':reasons}
    write_jsonl(out/'train_pairs.jsonl',all_pairs)
    summary={'protocol':protocol_name,'annotation_integrity':archive_integrity,
             'queries':len(records),'images':len(manifest_ids),'splits':dict(Counter(r['split'] for r in records)),
             'images_by_split':{s:len({r['image_id'] for r in records if r['split']==s}) for s in ('train','val','test')},
             'quarantined_by_split':dict(Counter(r['split'] for r in quarantine)),
             'rejected':dict(rejected),'pairs':pair_summary,
             'remote_image_bytes_read':remote.downloaded_bytes if remote else 0,
             'missing_requested_images':sorted(wanted-manifest_ids)}
    (out/'adapter_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
