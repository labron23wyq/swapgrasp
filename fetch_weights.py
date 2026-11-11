"""Fetch OpenAI CLIP's open_clip checkpoint through the HF mirror, with SHA."""
import json
from pathlib import Path
import requests
from download_data import download

endpoint='https://hf-mirror.com';repo='timm/vit_base_patch32_clip_224.openai'
revision='a6f597a30f7b82c51704746581f9a4e41421e878'
r=requests.get(f'{endpoint}/api/models/{repo}/tree/{revision}?recursive=true&limit=100',timeout=30);r.raise_for_status()
item=next(x for x in r.json() if x['path']=='open_clip_pytorch_model.bin');item['_url_prefix']=''
expected='9ecdaef325b20e7283dc6a32f92aa638d100899e4f084c2462d3832eeea0b26e'
if item.get('lfs',{}).get('oid')!=expected or item.get('size')!=605225782:
    raise ValueError('Pinned CLIP checkpoint metadata did not match the verified project weight')
download((endpoint,repo,revision,item,'weights/clip-vit-b32-openai.bin'))
print('OpenAI CLIP checkpoint ready',flush=True)
