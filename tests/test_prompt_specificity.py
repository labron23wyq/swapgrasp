import json

from scripts.report_submission import prompt_specificity


def test_specificity_separates_prompt_following_from_same_prediction(tmp_path):
    # A language-insensitive detector always proposes A; the second detector
    # proposes the requested target. All boxes are unambiguous and disjoint.
    a,b=[20,20,10,8,0],[80,20,10,8,0]
    rows=[{'sample_id':'a','image_id':'scene','split':'test','grasps':[a]},
          {'sample_id':'b','image_id':'scene','split':'test','grasps':[b]}]
    pair={'pair_id':'pair','image_id':'scene','split':'test','sample_a':'a','sample_b':'b'}
    manifest=tmp_path/'manifest.jsonl';pairs=tmp_path/'pairs.jsonl'
    manifest.write_text('\n'.join(json.dumps(row) for row in rows))
    pairs.write_text(json.dumps(pair))
    for method,boxes in [('baseline',[a,a]),('swap',[a,b])]:
        folder=tmp_path/f'{method}_s42';folder.mkdir()
        observations=[{'sample_id':sid,'predictions':[{'grasp':box,'score':1.}]} for sid,box in zip(('a','b'),boxes)]
        (folder/'test_predictions.jsonl').write_text('\n'.join(json.dumps(row) for row in observations))
    result=prompt_specificity(tmp_path,{'manifest':str(manifest),'pairs':str(pairs),'seeds':[42]})
    baseline=result['runs']['baseline_s42'];swap=result['runs']['swap_s42']
    assert baseline['target_at1']==baseline['cross_at1']==.5
    assert baseline['gap_at1']==0
    assert swap['target_at1']==1 and swap['cross_at1']==0
    assert swap['gap_at1']==1
    assert swap['directions']==2 and swap['pair_images']==1
    comparison=result['seed_comparisons']['42']['gap_at1']
    assert comparison['swap_minus_baseline']==1
    assert comparison['image_cluster_bootstrap_95ci']==[1,1]
