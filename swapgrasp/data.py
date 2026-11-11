"""Manifest-based data adapter and deterministic query/pair sampling."""
from collections import defaultdict,OrderedDict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from .geometry import transform_grasps


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def letterbox(image, size, nearest=False):
    """Use one isotropic affine transform, avoiding rounded anisotropic resize."""
    w, h = image.size
    scale = min(size/w, size/h)
    left, top = (size-scale*w)/2, (size-scale*h)/2
    transform = {'scale': scale, 'left': left, 'top': top,
                 'original_width': w, 'original_height': h, 'size': size}
    coeff = (1/scale, 0, -left/scale, 0, 1/scale, -top/scale)
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    out = image.transform((size, size), Image.Transform.AFFINE, coeff, resample=resample)
    support = Image.new('L', (w, h), 255).transform(
        (size, size), Image.Transform.AFFINE, coeff, resample=Image.Resampling.NEAREST)
    return out, np.asarray(support) > 0, transform


def center_support(grasps, height, width, radius=1):
    mask = np.zeros((height, width), dtype=bool)
    for x, y, *_ in grasps:
        x, y = int(math.floor(x+0.5)), int(math.floor(y+0.5))
        mask[max(0,y-radius):min(height,y+radius+1), max(0,x-radius):min(width,x+radius+1)] = True
    return mask


def load_target_mask(row, path):
    """Load an already-selected 2D per-target mask; reject ambiguous arrays."""
    path=Path(path)
    if path.suffix.lower()=='.npy':
        values=np.load(path,allow_pickle=False)
        if values.ndim==2:
            mask=values
        elif values.ndim==3 and row.get('mask_index') is not None:
            mask=values[int(row['mask_index'])]
        else:
            raise ValueError(f"Mask needs explicit 2D selection: {row['sample_id']}")
        return np.asarray(mask)>0
    with Image.open(path) as im:
        return np.asarray(im.convert('L'))>0


def make_targets(grasps, valid, sigma=2, radius=6, geom_radius=1):
    """Sparse whole-grasp alternatives at nearest rounded GT centres."""
    height, width = valid.shape
    quality = np.zeros((height, width), np.float32)
    centers = defaultdict(list)
    for g in grasps:
        x, y, w, h, theta = map(float, g)
        cx, cy = int(math.floor(x+0.5)), int(math.floor(y+0.5))
        if not (0 <= cx < width and 0 <= cy < height and valid[cy, cx]):
            raise ValueError('GT centre is outside valid letterboxed image')
        centers[(cx,cy)].append([math.cos(2*theta), math.sin(2*theta),
                                math.log(w/width), math.log(h/height)])
        x0, x1 = max(0,cx-radius), min(width,cx+radius+1)
        y0, y1 = max(0,cy-radius), min(height,cy+radius+1)
        yy, xx = np.mgrid[y0:y1,x0:x1]
        gaussian = np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma))
        quality[y0:y1,x0:x1] = np.maximum(quality[y0:y1,x0:x1], gaussian)
    nearest = {}
    for (cx,cy), alternatives in centers.items():
        for y in range(max(0,cy-geom_radius),min(height,cy+geom_radius+1)):
            for x in range(max(0,cx-geom_radius),min(width,cx+geom_radius+1)):
                if not valid[y,x]:
                    continue
                dist = (x-cx)**2+(y-cy)**2
                old = nearest.get((y,x))
                if old is None or dist < old[0]:
                    nearest[(y,x)] = [dist, list(alternatives)]
                elif dist == old[0]:
                    old[1].extend(alternatives)
    indexes, vectors, groups = [], [], []
    for group, ((y,x), (_, alternatives)) in enumerate(sorted(nearest.items())):
        for v in alternatives:
            indexes.append(y*width+x)
            vectors.append(v)
            groups.append(group)
    return {'quality': torch.from_numpy(quality), 'valid': torch.from_numpy(valid.copy()),
            'support': torch.from_numpy(center_support(grasps,height,width) & valid),
            'geom_index': torch.tensor(indexes, dtype=torch.long),
            'geom_vectors': torch.tensor(vectors, dtype=torch.float32).reshape(-1,4),
            'geom_group': torch.tensor(groups,dtype=torch.long), 'geom_count': len(nearest)}


class ManifestDataset:
    def __init__(self, manifest, split, size=320, root=None):
        self.manifest = Path(manifest).resolve()
        self.root = Path(root).resolve() if root else self.manifest.parent
        self.rows = [r for r in read_jsonl(manifest) if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No records in split {split}')
        self.by_id = {r['sample_id']: i for i,r in enumerate(self.rows)}
        if len(self.by_id) != len(self.rows):
            raise ValueError('Duplicate sample_id')
        self.size = size
        self.cache_images=0;self.cache_targets=0
        self._images=OrderedDict();self._targets=OrderedDict()

    def path(self, value):
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    def image(self, row):
        key=(row['image_id'],row['image'])
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        with Image.open(self.path(row['image'])) as im:
            image, valid, transform = letterbox(im.convert('RGB'), self.size)
        x = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
        x = (x-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
        result=x,valid,transform
        if self.cache_images:
            self._images[key]=result
            while len(self._images)>self.cache_images:self._images.popitem(last=False)
        return result

    def batch(self, ids, with_targets=True):
        # Inference path intentionally does not read mask or grasp fields.
        unique, images, valid, transforms, mapping, rows, targets = {}, [], [], [], [], [], []
        for i in ids:
            row = self.rows[i]
            key = (row['image_id'], row['image'])
            if key not in unique:
                if any(k[0]==key[0] and k[1]!=key[1] for k in unique):
                    raise ValueError(f"Image id maps to conflicting paths: {key[0]}")
                unique[key] = len(images)
                x, v, tr = self.image(row)
                images.append(x)
                valid.append(v)
                transforms.append(tr)
            j = unique[key]
            mapping.append(j)
            rows.append(row)
            if with_targets:
                if row['sample_id'] in self._targets:
                    self._targets.move_to_end(row['sample_id']);targets.append(self._targets[row['sample_id']]);continue
                gs = transform_grasps(row['grasps'], transforms[j])
                target_valid = valid[j].copy()
                if row.get('ignore_mask'):
                    ignored, _, _ = letterbox(Image.fromarray(load_target_mask(row,self.path(row['ignore_mask'])).astype(np.uint8)*255), self.size, nearest=True)
                    target_valid &= np.asarray(ignored) == 0
                targets.append(make_targets(gs, target_valid))
                if self.cache_targets:
                    self._targets[row['sample_id']]=targets[-1]
                    while len(self._targets)>self.cache_targets:self._targets.popitem(last=False)
        return {'images': torch.stack(images), 'mapping': torch.tensor(mapping),
                'texts': [r['text'] for r in rows], 'rows': rows, 'targets': targets,
                'valid': torch.from_numpy(np.stack([valid[j] for j in mapping])),
                'transforms': [transforms[j] for j in mapping]}


class PairSampler:
    def __init__(self, dataset, pair_file, queries=8, pairs_per_step=2, seed=42):
        self.dataset, self.queries, self.pairs_per_step = dataset, queries, pairs_per_step
        self.rng = random.Random(seed)
        rows = read_jsonl(pair_file) if pair_file else []
        self.pairs = [(dataset.by_id[p['sample_a']],dataset.by_id[p['sample_b']])
                      for p in rows if p['sample_a'] in dataset.by_id and p['sample_b'] in dataset.by_id]
        if pairs_per_step and not self.pairs:
            raise ValueError('No verified pairs for requested pair sampler')
        if 2*pairs_per_step > queries:
            raise ValueError('Pair queries exceed batch size')

    def next(self):
        ids = [self.rng.randrange(len(self.dataset.rows)) for _ in range(self.queries-2*self.pairs_per_step)]
        positions = []
        for _ in range(self.pairs_per_step):
            a,b = self.rng.choice(self.pairs)
            positions.append((len(ids),len(ids)+1))
            ids.extend([a,b])
        return ids, positions

    def state_dict(self):
        return {'rng': self.rng.getstate()}

    def load_state_dict(self, state):
        self.rng.setstate(state['rng'])


def build_pairs(dataset):
    """Verify mappings and geometry; language semantics need separate inspection."""
    by_image = defaultdict(list)
    for row in dataset.rows:
        by_image[row['image_id']].append(row)
    accepted, rejected = [], defaultdict(int)
    for image_id, rows in by_image.items():
        masks, supports = {}, {}
        for a,b in itertools.combinations(rows,2):
            if a['object_id'] == b['object_id']:
                rejected['same_object'] += 1
                continue
            if not all(r.get('mapping_verified',r.get('semantic_verified',False)) for r in [a,b]):
                rejected['unverified_mapping'] += 1
                continue
            if a.get('mask_level') not in ['object','part'] or a.get('mask_level')!=b.get('mask_level'):
                rejected['requires_same_granularity_masks'] += 1
                continue
            if a['text'].strip().casefold() == b['text'].strip().casefold():
                rejected['same_instruction'] += 1
                continue
            if not a.get('mask') or not b.get('mask'):
                rejected['missing_mask'] += 1
                continue
            for r in (a,b):
                sid=r['sample_id']
                if sid not in masks:masks[sid]=load_target_mask(r,dataset.path(r['mask']))
            ma,mb=(masks[r['sample_id']] for r in (a,b))
            if ma.shape != mb.shape or not ma.any() or not mb.any() or np.any(ma & mb):
                rejected['empty_or_overlapping_masks'] += 1
                continue
            membership = True
            for r, mask in [(a,ma),(b,mb)]:
                for x,y,*_ in r['grasps']:
                    x,y = int(math.floor(x+.5)),int(math.floor(y+.5))
                    membership &= 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y,x]
            if not membership:
                rejected['grasp_center_outside_target'] += 1
                continue
            pending=[r['sample_id'] for r in (a,b) if r['sample_id'] not in supports]
            if pending:
                batch=dataset.batch([dataset.by_id[sid] for sid in pending])
                supports.update((sid,t['support'].numpy()) for sid,t in zip(pending,batch['targets']))
            pa,pb=(supports[r['sample_id']] for r in (a,b))
            if not pa.any() or not pb.any() or (pa & pb).any():
                rejected['overlapping_grasp_support'] += 1
                continue
            key = a['sample_id']+'|'+b['sample_id']
            accepted.append({'pair_id': hashlib.sha256(key.encode()).hexdigest()[:16],
                             'image_id': image_id, 'sample_a': a['sample_id'],
                             'sample_b': b['sample_id'], 'split': a['split'],
                             'pair_level': 'different_objects_'+a['mask_level']+'_supports', 'mask_overlap': 0,
                             'verification_scope':'filename_mapping_disjoint_masks_and_grasp_supports',
                             'semantic_review_required':not all(r.get('semantic_verified',False) for r in [a,b])})
    return accepted, dict(rejected)

