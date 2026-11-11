import math
from swapgrasp.metrics import extended_scores


def test_extended_metrics_keep_standard_and_measure_pairs_and_geometry():
    rows=[{'sample_id':'a','image_id':'im','category_split':'seen',
           'grasps':[[10.,10.,8.,4.,0.]]},
          {'sample_id':'b','image_id':'im','category_split':'unseen',
           'grasps':[[30.,10.,8.,4.,0.]]}]
    predictions=[{'sample_id':'a','image_id':'im','predictions':[{'grasp':[10.,10.,8.,4.,0.],'score':.9}]},
                 {'sample_id':'b','image_id':'im','predictions':[{'grasp':[30.,10.,8.,4.,math.radians(20)],'score':.9}]}]
    pairs=[{'sample_a':'a','sample_b':'b','image_id':'im'}]
    metrics,per_query=extended_scores(rows,predictions,pairs)
    assert metrics['standard_metrics']['grasp_at1']==1.
    assert metrics['thresholds']['standard']['at1']==1.
    assert metrics['thresholds']['strict']['at1']==.5
    assert metrics['thresholds']['very_strict']['at1']==.5
    assert metrics['thresholds']['strict']['pair_both_at1']==0.
    assert abs(per_query[1]['top1_geometry']['angle_degrees']-20.) < .01


def test_extended_metrics_do_not_mix_gt_and_respect_candidate_rank():
    row={'sample_id':'a','image_id':'im','grasps':[[10.,10.,8.,4.,math.pi/2],[100.,100.,8.,4.,0.]]}
    predictions=[{'sample_id':'a','image_id':'im','predictions':[
        {'grasp':[10.,10.,8.,4.,0.],'score':.9},
        {'grasp':[100.,100.,8.,4.,0.],'score':.8}]}]
    metrics,_=extended_scores([row],predictions)
    assert metrics['thresholds']['standard']['at1']==0.
    assert metrics['thresholds']['standard']['at5']==1.
    assert metrics['thresholds']['very_strict']['at5']==1.


def test_missing_predictions_are_failures_and_have_no_geometric_error():
    row={'sample_id':'a','image_id':'im','grasps':[[10.,10.,8.,4.,0.]]}
    metrics,per_query=extended_scores([row],[{'sample_id':'a','image_id':'im','predictions':[]}])
    assert metrics['thresholds']['strict']['at1']==0.
    assert metrics['thresholds']['strict']['pair_both_at1'] is None
    assert metrics['top1_geometry_count']==0 and per_query[0]['top1_geometry'] is None
