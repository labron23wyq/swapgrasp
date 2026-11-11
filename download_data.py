"""Download authorized official HF datasets through an optional transport mirror.

No credentials are stored. LFS SHA-256 is verified before promotion from .part.
Range requests are bounded and Content-Range is checked before appending.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import requests

FILES = {
    'airvlab/Grasp-Anything': ['image_part_aa','image_part_ab','scene_description.zip','mask.zip'],
    'airvlab/Grasp-Anything-pp': ['grasp_instructions.zip','grasp_label_positive.zip','part_mask.zip'],
}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def download(job):
    endpoint,repo,revision,item,dest=job
    expected,oid=item['size'],item['lfs']['oid']
    dest=Path(dest); dest.parent.mkdir(parents=True,exist_ok=True)
    part=dest.with_name(dest.name+'.part')
    if dest.exists():
        if dest.stat().st_size==expected and sha256(dest)==oid:
            print('VERIFIED existing',dest,flush=True)
            return
        raise ValueError(f'Existing file failed verification: {dest}')
    if part.exists() and part.stat().st_size>expected:
        raise ValueError(f'Oversized partial file {part}')
    start=time.monotonic(); initial=part.stat().st_size if part.exists() else 0
    failures=0
    while (part.stat().st_size if part.exists() else 0)<expected:
        offset=part.stat().st_size if part.exists() else 0
        end=min(expected-1,offset+64*1024*1024-1)
        # Query distinguishes cached responses to different ranges at proxies.
        prefix=item.get('_url_prefix','datasets/')
        url=f'{endpoint}/{prefix}{repo}/resolve/{revision}/{item["path"]}?download=true&offset={offset}'
        try:
            with requests.get(url,headers={'Range':f'bytes={offset}-{end}'},stream=True,timeout=(20,90)) as r:
                r.raise_for_status()
                cr=r.headers.get('Content-Range','')
                if r.status_code!=206 or cr!=f'bytes {offset}-{end}/{expected}':
                    raise ValueError(f'Unexpected range response {r.status_code}: {cr}')
                written=0
                with open(part,'ab') as f:
                    for chunk in r.iter_content(1024*1024):
                        if chunk:
                            if written+len(chunk)>end-offset+1:
                                raise ValueError('Server returned beyond requested byte range')
                            f.write(chunk); written+=len(chunk)
                if written!=end-offset+1:
                    raise IOError('Truncated range response')
            failures=0
            current=part.stat().st_size
            speed=(current-initial)/max(time.monotonic()-start,1)/1e6
            print(f'DOWNLOAD {repo}/{item["path"]} {current}/{expected} {current/expected:.1%} {speed:.1f} MB/s',flush=True)
        except (requests.RequestException,IOError,ValueError) as e:
            failures+=1
            print('RETRY',repo,item['path'],failures,type(e).__name__,str(e)[:160],flush=True)
            if failures>=12:
                raise
            time.sleep(min(2*failures,20))
    if sha256(part)!=oid:
        raise ValueError(f'SHA256 mismatch: {part}; retained for investigation')
    part.replace(dest)
    dest.with_name(dest.name+'.verified.json').write_text(json.dumps({
        'repo':repo,'revision':revision,'path':item['path'],'size':expected,'sha256':oid,
        'canonical_url':f'https://huggingface.co/{item.get("_url_prefix","datasets/")}{repo}/resolve/{revision}/{item["path"]}',
        'transport_endpoint':endpoint},indent=2))
    print('VERIFIED',dest,flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',default='data/downloads')
    p.add_argument('--endpoint',default='https://hf-mirror.com'); p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    jobs=[]; catalog={}
    for repo,files in FILES.items():
        r=requests.get(f'{a.endpoint}/api/datasets/{repo}',timeout=30); r.raise_for_status()
        revision=r.json()['sha']
        r=requests.get(f'{a.endpoint}/api/datasets/{repo}/tree/{revision}?recursive=true&limit=1000',timeout=30); r.raise_for_status()
        tree={x['path']:x for x in r.json()}; catalog[repo]={'revision':revision,'files':tree}
        for name in files:
            jobs.append((a.endpoint,repo,revision,tree[name],str(root/repo.split('/')[-1]/name)))
    (root/'catalog.json').write_text(json.dumps(catalog,indent=2))
    # Smaller annotation files start first; images also download in parallel.
    jobs.sort(key=lambda j:j[3]['size'])
    print('TOTAL bytes',sum(j[3]['size'] for j in jobs),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(download,j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):
            f.result()


if __name__=='__main__':
    main()
