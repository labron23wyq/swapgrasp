import hashlib
import math
from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.ops import FeaturePyramidNetwork


def resize_bilinear(x, size):
    """align_corners=False bilinear interpolation via deterministic GEMMs.

    CUDA's native bilinear backward uses unordered atomic accumulation. The
    separable matrix version gives the same interpolation rule and deterministic
    backward under PyTorch deterministic algorithms / cuBLAS workspace config.
    """
    if x.shape[-2:]==tuple(size):
        return x
    def weights(n_in,n_out):
        pos=(torch.arange(n_out,device=x.device,dtype=torch.float32)+.5)*(n_in/n_out)-.5
        lo=pos.floor().long(); fraction=pos-lo
        columns=torch.arange(n_in,device=x.device)[None,:]
        matrix=(columns==lo.clamp(0,n_in-1)[:,None])*(1-fraction[:,None])
        matrix=matrix+(columns==(lo+1).clamp(0,n_in-1)[:,None])*fraction[:,None]
        return matrix.to(x.dtype)
    wy,wx=weights(x.shape[-2],size[0]),weights(x.shape[-1],size[1])
    x=F.linear(x,wx)
    return F.linear(x.transpose(-1,-2),wy).transpose(-1,-2).contiguous()


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False),
                         nn.GroupNorm(min(8,cout),cout), nn.ReLU(inplace=True))


class SwapGrasp(nn.Module):
    def __init__(self, channels=128, pretrained=True, text_dim=512, image_weights_path=None):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained and not image_weights_path else None)
        if image_weights_path:
            net.load_state_dict(torch.load(image_weights_path,map_location='cpu',weights_only=True))
        self.stem = nn.Sequential(net.conv1,net.bn1,net.relu,net.maxpool)
        self.stages = nn.ModuleList([net.layer1,net.layer2,net.layer3,net.layer4])
        self.fpn = FeaturePyramidNetwork([64,128,256,512],channels)
        self.film = nn.ModuleList([nn.Sequential(nn.Linear(text_dim,256),nn.ReLU(),nn.Linear(256,2*channels)) for _ in range(4)])
        for layer in self.film:
            nn.init.zeros_(layer[-1].weight)
            nn.init.zeros_(layer[-1].bias)
        self.fusion = ConvBlock(4*channels,channels)
        self.up1,self.up2 = ConvBlock(channels,channels//2),ConvBlock(channels//2,channels//4)
        self.head = nn.Conv2d(channels//4,5,1)
        nn.init.normal_(self.head.weight,std=.001)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([-2.19,1.,0.,math.log(.15),math.log(.08)]))
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Freeze running statistics; affine backbone BN parameters remain trainable.
        for part in [self.stem,self.stages]:
            for module in part.modules():
                if isinstance(module,nn.BatchNorm2d):
                    module.eval()
        return self

    def forward(self, images, text_embeddings, image_index=None, return_features=False):
        x,features = self.stem(images),{}
        for i,stage in enumerate(self.stages):
            x = stage(x)
            features[str(i)] = x
        pyramid = self.fpn(features)
        if image_index is None:
            image_index = torch.arange(len(images),device=images.device)
        target_size = next(iter(pyramid.values())).shape[-2:]
        fused = []
        for i,p in enumerate(pyramid.values()):
            gamma,beta = self.film[i](text_embeddings).chunk(2,dim=1)
            conditioned = p[image_index]*(1+gamma[:,:,None,None])+beta[:,:,None,None]
            fused.append(resize_bilinear(conditioned,target_size))
        x = self.fusion(torch.cat(fused,dim=1))
        x = self.up1(resize_bilinear(x,(x.shape[-2]*2,x.shape[-1]*2)))
        x = self.up2(resize_bilinear(x,images.shape[-2:]))
        prediction = self.head(x)
        return (prediction,x) if return_features else prediction

class TextEncoder:
    def __init__(self, device, mode='clip', model='ViT-B-32', pretrained='openai'):
        self.device,self.mode,self.cache = device,mode,OrderedDict()
        self.max_cache_entries=50000
        if mode == 'clip':
            import open_clip
            self.net = open_clip.create_model(model,pretrained=pretrained,force_quick_gelu=True).to(device).eval()
            # We use only the text tower; release unused vision parameters.
            if hasattr(self.net,'visual'):
                del self.net.visual
            self.net.requires_grad_(False)
            self.tokenizer = open_clip.get_tokenizer(model)
        elif mode != 'debug_hash':
            raise ValueError(f'Unknown text encoder {mode}')

    def __call__(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if self.mode == 'debug_hash':
            # Deliberately has NO pretrained semantic ability. Synthetic smoke only.
            for text in missing:
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'little') % (2**63-1)
                g = torch.Generator().manual_seed(seed)
                self.cache[text] = F.normalize(torch.randn(512,generator=g),dim=0)
        elif missing:
            # CLIP tokenizer otherwise silently truncates at 77 tokens.
            if not hasattr(self.tokenizer,'encode'):
                raise TypeError('Expected inspectable CLIP tokenizer; audit truncation before use')
            for text in missing:
                if len(self.tokenizer.encode(text)) > 75:
                    raise ValueError(f'Instruction exceeds CLIP context: {text!r}')
            tokens = self.tokenizer(missing).to(self.device)
            with torch.inference_mode():
                embedding = self.net.encode_text(tokens,normalize=True).float().cpu()
            self.cache.update(zip(missing,embedding))
        for text in texts:
            self.cache.move_to_end(text)
        while len(self.cache)>self.max_cache_entries:
            self.cache.popitem(last=False)
        # Materialize ordinary tensors: FiLM's trainable linear layers need to
        # save text inputs for backward, which inference tensors cannot do.
        return torch.stack([self.cache[t] for t in texts]).to(self.device).clone()
