from pathlib import Path
import copy
import math
import numpy as np
import pytest
import torch
import yaml
from tools import synthetic,audit
from swapgrasp.data import ManifestDataset,PairSampler,build_pairs,make_targets,write_jsonl
from swapgrasp.losses import geometry_loss,swap_loss
from swapgrasp.model import SwapGrasp,TextEncoder


def test_sampler_exact_resume_and_shared_visual_features(tmp_path):
    synthetic(tmp_path)
    ds=ManifestDataset(tmp_path/'manifest.jsonl','train',64)
    sampler=PairSampler(ds,tmp_path/'pairs.jsonl',4,1,42)
    sampler.next();state=sampler.state_dict();expected=sampler.next()
    fresh=PairSampler(ds,tmp_path/'pairs.jsonl',4,1,42);fresh.load_state_dict(state)
    assert fresh.next()==expected
    pair=sampler.pairs[0]; batch=ds.batch(pair)
    assert len(batch['images'])==1 and len(batch['texts'])==2
    assert batch['mapping'].tolist()==[0,0]


def test_pair_builder_rejects_unverified_and_overlapping_targets(tmp_path):
    synthetic(tmp_path);ds=ManifestDataset(tmp_path/'manifest.jsonl','train',64)
    accepted,_=build_pairs(ds);assert len(accepted)==4
    ds.rows[0]['semantic_verified']=False
    accepted,rejected=build_pairs(ds);assert len(accepted)==3
    assert rejected['unverified_mapping']>0
    ds.rows[2]['mask']=ds.rows[3]['mask']
    _,rejected=build_pairs(ds);assert rejected['empty_or_overlapping_masks']>0


def test_prediction_view_does_not_access_grasps_or_masks(tmp_path):
    synthetic(tmp_path);ds=ManifestDataset(tmp_path/'manifest.jsonl','test',64)
    for row in ds.rows:
        del row['grasps'];row['mask']='DOES_NOT_EXIST.npy'
    batch=ds.batch([0,1],with_targets=False)
    assert batch['targets']==[] and len(batch['images'])==1
    from swapgrasp.visualization import PredictionExporter
    exporter=PredictionExporter(tmp_path/'visuals')
    exporter.add(ds,batch['rows'][0],[],np.zeros((64,64),np.float32),batch['transforms'][0])
    exporter.finish()
    assert len(list((tmp_path/'visuals').glob('*.png')))==2
    assert (tmp_path/'visuals/index.html').exists()


def test_training_caches_preserve_supervision_and_inference_stays_label_free(tmp_path):
    synthetic(tmp_path);ds=ManifestDataset(tmp_path/'manifest.jsonl','train',64)
    reference=ds.batch([0,1])
    ds.cache_images=2;ds.cache_targets=2
    ds.batch([0,1]);cached=ds.batch([0,1])
    assert torch.equal(reference['images'],cached['images'])
    for original,target in zip(reference['targets'],cached['targets']):
        for key,value in original.items():
            assert (torch.equal(value,target[key]) if torch.is_tensor(value) else value==target[key])
    for row in ds.rows:row.pop('grasps',None)
    assert ds.batch([0,1],with_targets=False)['targets']==[]


def test_structural_audit_finds_image_leakage(tmp_path):
    synthetic(tmp_path);ds=ManifestDataset(tmp_path/'manifest.jsonl','train',64)
    rows=ds.rows;rows[1]['split']='val';write_jsonl(tmp_path/'bad.jsonl',rows)
    assert not audit(tmp_path/'bad.jsonl',tmp_path,tmp_path/'audit.json')


def test_whole_vector_min_rejects_channelwise_chimera():
    # Each channel individually occurs in a label, but the combined vector does not.
    pred=torch.tensor([[[1.]],[[0.]],[[1.]],[[1.]]],requires_grad=True)
    target={'geom_count':1,'geom_index':torch.tensor([0,0]),
            'geom_vectors':torch.tensor([[1.,0.,0.,0.],[0.,0.,1.,1.]]),
            'geom_group':torch.tensor([0,0])}
    loss=geometry_loss(pred,target);assert loss>.1;loss.backward();assert torch.isfinite(pred.grad).all()


def test_empty_pair_loss_is_differentiable_zero():
    z=torch.randn(2,8,8,requires_grad=True);empty=torch.zeros(8,8,dtype=torch.bool)
    loss=swap_loss(z,[empty,empty],[(0,1)]);assert loss.item()==0
    loss.backward();assert z.grad is not None and z.grad.abs().sum()==0


def test_network_shapes_and_trainable_text_projection(tmp_path):
    torch.set_num_threads(2)
    net=SwapGrasp(channels=32,pretrained=False)
    text=TextEncoder('cpu',mode='debug_hash')(['a','b'])
    x=torch.randn(1,3,64,64);y=net(x,text,torch.tensor([0,0]))
    assert y.shape==(2,5,64,64)
    y.mean().backward()
    assert all(layer[-1].weight.grad is not None for layer in net.film)
    assert all(not m.training for m in net.modules() if isinstance(m,torch.nn.BatchNorm2d))


def test_baseline_method_configs_only_change_swap_weight():
    root=Path(__file__).resolve().parents[1]
    a=yaml.safe_load((root/'configs/baseline.yaml').read_text());b=yaml.safe_load((root/'configs/swap.yaml').read_text())
    assert a['loss']['lambda_swap']==0 and b['loss']['lambda_swap']>0
    b['loss']['lambda_swap']=a['loss']['lambda_swap'];assert a==b


def test_official_angle_conversion_matches_official_vertices():
    # Official uses angle=-source_degrees and y-components -sin(angle).
    from swapgrasp.geometry import corners
    for deg in [-70,-20,0,45,89]:
        theta=-math.radians(deg); x,y,w,h=20,30,15,8
        xo,yo=math.cos(theta),math.sin(theta)
        y1,x1=y+w/2*yo,x-w/2*xo
        y2,x2=y-w/2*yo,x+w/2*xo
        official=np.array([[x1-h/2*yo,y1-h/2*xo],[x2-h/2*yo,y2-h/2*xo],
                           [x2+h/2*yo,y2+h/2*xo],[x1+h/2*yo,y1+h/2*xo]])
        assert np.allclose(official,corners([x,y,w,h,math.radians(deg)]),atol=1e-5)


def test_deterministic_bilinear_matches_native_forward_and_gradient():
    from swapgrasp.model import resize_bilinear
    import torch.nn.functional as F
    x=torch.randn(2,3,5,7,requires_grad=True)
    a=resize_bilinear(x,(13,18))
    b=F.interpolate(x,size=(13,18),mode='bilinear',align_corners=False)
    assert torch.allclose(a,b,atol=2e-6)
    grad=torch.randn_like(a)
    ga=torch.autograd.grad(a,x,grad,retain_graph=True)[0]
    gb=torch.autograd.grad(b,x,grad)[0]
    assert torch.allclose(ga,gb,atol=3e-6)

