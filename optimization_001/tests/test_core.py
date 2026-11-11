import math
import numpy as np
import torch
from swapgrasp.geometry import angle_error,corners,decode,grasp_success,rotated_iou,transform_grasps
from swapgrasp.data import letterbox,make_targets,transform_grasps as transform
from swapgrasp.losses import geometry_loss,region_response,swap_loss


def test_grasp_roundtrip_and_pi_period():
    for theta in [-1.4,-.7,0,.4,1.3]:
        g=[30,20,18,8,theta]
        assert np.allclose(corners(g).mean(0),g[:2],atol=1e-5)
        assert angle_error(theta,theta+math.pi)<1e-7
        assert rotated_iou(g,[30,20,18,8,theta+math.pi])>.99
    assert angle_error(math.radians(179),math.radians(-1))<1e-6


def test_letterbox_preserves_geometry_and_roundtrips():
    from PIL import Image
    image=Image.new('RGB',(200,100))
    _,valid,tr=letterbox(image,320)
    assert valid.sum()==160*320
    gs=np.array([[80,40,30,10,.4]],dtype=np.float32)
    resized=transform(gs,tr)
    restored=transform_grasps(resized,tr,inverse=True)
    assert np.allclose(gs,restored,atol=1e-5)


def test_swap_is_symmetric_and_gradients_point_toward_both_targets():
    z=torch.zeros(2,12,12,requires_grad=True)
    pa=torch.zeros(12,12,dtype=torch.bool); pa[2:4,2:4]=True
    pb=torch.zeros_like(pa); pb[8:10,8:10]=True
    correct=swap_loss(z,[pa,pb],[(0,1)])
    correct.backward(); grad=z.grad.clone(); z.grad.zero_()
    assert correct.item()>0
    assert grad[0][pa].sum()<0 and grad[0][pb].sum()>0
    assert grad[1][pb].sum()<0 and grad[1][pa].sum()>0
    swapped=swap_loss(z.detach().flip(0).requires_grad_(),[pa,pb],[(0,1)])
    assert torch.allclose(correct.detach(),swapped.detach())


def test_region_pooling_nonempty_and_numerically_stable():
    z=torch.tensor([[1000.,999.]])
    support=torch.tensor([[True,True]])
    score=region_response(z,support,.25)
    assert torch.isfinite(score) and 999 < score < 1000


def test_geometry_uses_one_complete_valid_alternative():
    pred=torch.tensor([[[1.]],[[0.]],[[0.]],[[0.]]])
    target={'geom_count':1,'geom_index':torch.tensor([0,0]),
            'geom_vectors':torch.tensor([[1.,0.,0.,0.],[0.,0.,1.,1.]]),
            'geom_group':torch.tensor([0,0])}
    assert geometry_loss(pred,target)<1e-5


def test_decode_uses_prediction_only_and_valid_region():
    out=torch.zeros(5,32,32); out[0,10,12]=8; out[1,10,12]=1; out[3,10,12]=math.log(.5);out[4,10,12]=math.log(.2)
    valid=torch.ones(32,32,dtype=torch.bool); valid[:5]=False
    ps=decode(out,valid)
    assert ps and ps[0]['grasp'][:2]==[12.,10.]
    assert abs(ps[0]['grasp'][2]-16)<1e-5


def test_same_grasp_must_meet_both_success_conditions():
    assert grasp_success([{'grasp':[10,10,8,4,0]}],[[10,10,8,4,0]])
    assert not grasp_success([{'grasp':[10,10,8,4,math.pi/3]}],[[10,10,8,4,0]])

