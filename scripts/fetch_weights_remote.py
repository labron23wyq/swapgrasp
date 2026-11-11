import os,json,pathlib,torch
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from torchvision.models import resnet18,ResNet18_Weights
print('FETCH RESNET18',flush=True)
r=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print('RESNET18 READY',flush=True)
import open_clip
print('FETCH CLIP',open_clip.get_pretrained_cfg('ViT-B-32','openai'),flush=True)
clip=open_clip.create_model('ViT-B-32',pretrained='openai')
tokenizer=open_clip.get_tokenizer('ViT-B-32')
print('CLIP TOKENIZER',type(tokenizer).__name__,hasattr(tokenizer,'encode'),flush=True)
pathlib.Path('artifacts/pretrained_ready.json').write_text(json.dumps({'image':'ResNet18_IMAGENET1K_V1','text':'ViT-B-32/openai'},indent=2))
print('CLIP READY',flush=True)
