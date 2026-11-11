import copy
import torch
from tools import synthetic
from swapgrasp.data import ManifestDataset,PairSampler
from swapgrasp.losses import hard_swap_loss,swap_weight
from swapgrasp.model import SwapGrasp


def test_balanced_sampler_resume_and_pair_indices(tmp_path):
    synthetic(tmp_path);ds=ManifestDataset(tmp_path/'manifest.jsonl','train',64)
    sampler=PairSampler(ds,tmp_path/'pairs.jsonl',queries=16,pairs_per_step=1,image_balanced=True)
    sampler.next();state=copy.deepcopy(sampler.state_dict());expected=sampler.next()
    sampler.load_state_dict(state);assert sampler.next()==expected
    ids,pairs=expected
    assert len(ids)==16 and len(pairs)==1
    a,b=pairs[0];assert ds.rows[ids[a]]['image_id']==ds.rows[ids[b]]['image_id']


def test_hard_ranking_gradients_and_satisfied_margin():
    a=torch.tensor([[True,False]]);b=~a
    logits=torch.zeros(2,1,2,requires_grad=True)
    loss,stats=hard_swap_loss(logits,[a,b],[(0,1)])
    loss.backward();assert logits.grad[0,0,0]<0 and logits.grad[0,0,1]>0
    assert stats['active_fraction']==1
    solved=torch.tensor([[[2.,-2.]],[[-2.,2.]]],requires_grad=True)
    loss,stats=hard_swap_loss(solved,[a,b],[(0,1)])
    assert loss==0 and stats['active_fraction']==0
    loss.backward();assert solved.grad.abs().sum()==0


def test_swap_schedule_and_optional_feature_output():
    cfg={'lambda_swap':.05,'swap_start':500,'swap_ramp':500}
    assert swap_weight(500,cfg)==0 and swap_weight(750,cfg)==.025 and swap_weight(1000,cfg)==.05
    torch.set_num_threads(2);model=SwapGrasp(32,False).eval()
    image=torch.randn(1,3,64,64);text=torch.randn(2,512);mapping=torch.tensor([0,0])
    with torch.no_grad():
        prediction,features=model(image,text,mapping,return_features=True)
        original=model(image,text,mapping)
    assert torch.equal(prediction,original) and features.shape==(2,8,64,64)


def test_reranking_keeps_candidate_geometry_and_zero_weight_is_baseline():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('rerank',Path(__file__).resolve().parents[1]/'scripts/rerank_candidates.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cache={'base_logits':torch.tensor([[2.,1.,0.,0.,0.]]),
           'valid':torch.tensor([[True,True,False,False,False]]),
           'proposals':[{'sample_id':'s','image_id':'i','predictions':[
               {'grasp':[10.,10.,4.,2.,0.],'score':.8},
               {'grasp':[20.,20.,4.,2.,0.],'score':.7}]}]}
    residual=torch.zeros(1,5,2);residual[0,1,0]=10
    assert module.reordered(cache,residual,0)==cache['proposals']
    ranked=module.reordered(cache,residual,1)
    assert ranked[0]['predictions']==list(reversed(cache['proposals'][0]['predictions']))
    class Dataset:
        rows=[{'sample_id':'s','grasps':[[20.,20.,4.,2.,0.]]}]
    labels,quality=module.labels(Dataset(),cache)
    assert labels[0,0].sum()==0 and labels[0,1].tolist()==[1.,1.]
    assert quality[0,1]>.99


def test_candidate_feature_extraction_reads_no_labels(tmp_path):
    import importlib.util
    from pathlib import Path
    import numpy as np
    from swapgrasp.model import TextEncoder
    from train import predict
    spec=importlib.util.spec_from_file_location('rank_extract',Path(__file__).resolve().parents[1]/'scripts/rerank_candidates.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    synthetic(tmp_path);dataset=ManifestDataset(tmp_path/'manifest.jsonl','test',64)
    for row in dataset.rows:row.pop('grasps');row['mask']='nonexistent.npy'
    torch.set_num_threads(2);model=SwapGrasp(32,False);text=TextEncoder('cpu',mode='debug_hash')
    cache=module.extract(model,text,dataset,torch.device('cpu'),tmp_path/'cache.pt')
    expected=predict(model,text,dataset,torch.device('cpu'),32)
    assert cache['features'].shape==(len(dataset.rows),5,599)
    for a,b in zip(cache['proposals'],expected):
        for p,q in zip(a['predictions'],b['predictions']):
            assert np.allclose(p['grasp'],q['grasp']) and abs(p['score']-q['score'])<1e-6
