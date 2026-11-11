from collections import defaultdict
import math
import numpy as np
from .geometry import angle_error, grasp_success, rotated_iou


THRESHOLDS = {'standard': (0.25, 30), 'strict': (0.50, 15), 'very_strict': (0.75, 10)}


def extended_scores(rows, predictions, pairs=()):
    """Supplementary offline metrics on the same ranked candidates and GTs."""
    primary, base = score_predictions(rows, predictions, pairs)
    by_id = {p['sample_id']: p['predictions'] for p in predictions}
    grid = sorted(set(THRESHOLDS.values()) | {(i, a) for i in (.25,.35,.50,.65,.75) for a in (10,15,20,30)})
    matches, geometry, per_query = {}, [], []
    for row, original in zip(rows, base):
        sid = row['sample_id']
        gt = row['grasps']
        pred = [p['grasp'] for p in by_id[sid][:5]]
        ious = np.array([[rotated_iou(p,g) for g in gt] for p in pred]).reshape(len(pred),len(gt))
        angles = np.array([[angle_error(p[4],g[4]) for g in gt] for p in pred]).reshape(ious.shape)
        matches[sid] = {}
        for iou, deg in grid:
            hits = (ious > iou) & (angles < math.radians(deg))
            matches[sid][iou,deg] = (bool(hits[:1].any()), bool(hits.any()))
        # Compare all geometric components to one reference: highest-IoU GT,
        # with nearest centre breaking ties (including all-zero IoUs).
        error = None
        if pred and gt:
            p = pred[0]
            j = min(range(len(gt)), key=lambda j: (-ious[0,j],math.hypot(p[0]-gt[j][0],p[1]-gt[j][1])))
            g = gt[j]
            error = {'iou':float(ious[0,j]), 'angle_degrees':math.degrees(angles[0,j]),
                     'center_px':math.hypot(p[0]-g[0],p[1]-g[1]),
                     'relative_width_error':abs(p[2]-g[2])/g[2],
                     'relative_height_error':abs(p[3]-g[3])/g[3]}
            geometry.append(error)
        result = {**original,'top1_geometry':error}
        for name, threshold in THRESHOLDS.items():
            at1,at5 = matches[sid][threshold]
            result.update({name+'_at1':int(at1),name+'_at5':int(at5)})
        if (result['standard_at1'],result['standard_at5'])!=(original['grasp_at1'],original['grasp_at5']):
            raise ValueError('Extended scorer disagrees with the standard success definition')
        per_query.append(result)
    valid = [(p['sample_a'],p['sample_b'],p['image_id']) for p in pairs
             if p['sample_a'] in matches and p['sample_b'] in matches]
    images = defaultdict(list)
    for r in rows:images[r['image_id']].append(r['sample_id'])
    table = []
    for threshold in grid:
        hits = [m[threshold] for m in matches.values()]
        item = {'iou_gt':threshold[0], 'angle_lt_degrees':threshold[1]}
        for idx,k in enumerate((1,5)):
            values = {sid:m[threshold][idx] for sid,m in matches.items()}
            count = sum(h[idx] for h in hits)
            both = sum(values[a] and values[b] for a,b,_ in valid)
            item.update({f'count_at{k}':count, f'at{k}':count/len(rows),
                         f'image_macro_at{k}':float(np.mean([np.mean([values[s] for s in ids]) for ids in images.values()])),
                         f'pair_count_both_at{k}':both, f'pair_both_at{k}':both/len(valid) if valid else None})
        item['groups']={}
        for group in ('seen','unseen'):
            ids=[r['sample_id'] for r in rows if r.get('category_split')==group]
            item['groups'][group]={'queries':len(ids),'at1':sum(matches[s][threshold][0] for s in ids)/len(ids) if ids else None}
        table.append(item)
    geo = {k:dict(zip(('p25','median','p75','p90'),np.quantile([g[k] for g in geometry],[.25,.5,.75,.9]).tolist()))
           for k in geometry[0]} if geometry else {}
    return {'standard_metrics':primary,'pair_images':len({i for _,_,i in valid}),
            'thresholds':{k:next(t for t in table if (t['iou_gt'],t['angle_lt_degrees'])==v) for k,v in THRESHOLDS.items()},
            'threshold_grid':table,'top1_geometry_count':len(geometry),'top1_geometry':geo}, per_query


def score_predictions(rows, predictions, pairs=None):
    by_id={p['sample_id']:p for p in predictions}
    if len(by_id)!=len(predictions):
        raise ValueError('Duplicate prediction id')
    if set(by_id)!={r['sample_id'] for r in rows}:
        raise ValueError('Prediction IDs do not exactly match requested evaluation split')
    per_query=[]
    for row in rows:
        pred=by_id[row['sample_id']]
        if pred['image_id']!=row['image_id']:
            raise ValueError('Prediction image_id mismatch')
        result={'sample_id':row['sample_id'],'image_id':row['image_id'],
                'category_split':row.get('category_split','unspecified')}
        for k in [1,5]:
            result[f'grasp_at{k}']=int(grasp_success(pred['predictions'],row['grasps'],k=k))
        per_query.append(result)
    grouped=defaultdict(list)
    for r in per_query:
        grouped[r['image_id']].append(r)
    metrics={'queries':len(rows),'images':len(grouped)}
    for k in [1,5]:
        key=f'grasp_at{k}'
        metrics[key]=float(np.mean([r[key] for r in per_query]))
        metrics[key+'_image_macro']=float(np.mean([np.mean([r[key] for r in rs]) for rs in grouped.values()]))
    query_map={r['sample_id']:r for r in per_query}
    valid_pairs=[p for p in (pairs or []) if p['sample_a'] in query_map and p['sample_b'] in query_map]
    metrics['pair_count']=len(valid_pairs)
    metrics['pair_images']=len({p['image_id'] for p in valid_pairs})
    metrics['pair_both_grasp_at1']=(float(np.mean([
        query_map[p['sample_a']]['grasp_at1']*query_map[p['sample_b']]['grasp_at1'] for p in valid_pairs]))
        if valid_pairs else None)
    for group in ['seen','unseen']:
        values=[r['grasp_at1'] for r in per_query if r['category_split']==group]
        metrics[group+'_count']=len(values)
        metrics[group+'_grasp_at1']=float(np.mean(values)) if values else None
    s,u=metrics['seen_grasp_at1'],metrics['unseen_grasp_at1']
    metrics['seen_unseen_hmean']=None if s is None or u is None else (2*s*u/(s+u) if s+u else 0.)
    return metrics,per_query


def paired_bootstrap(rows_a, rows_b, replicates=1000, seed=42, metric='grasp_at1'):
    """Query-micro difference, resampling whole images with all their queries."""
    a={r['sample_id']:r for r in rows_a}; b={r['sample_id']:r for r in rows_b}
    if set(a)!=set(b):
        raise ValueError('Paired comparison requires identical query IDs')
    images=defaultdict(list)
    for key,ra in a.items():
        if ra['image_id']!=b[key]['image_id']:
            raise ValueError('Image mismatch')
        images[ra['image_id']].append(b[key][metric]-ra[metric])
    arrays=list(images.values()); rng=np.random.default_rng(seed); values=[]
    for _ in range(replicates):
        sample=rng.integers(0,len(arrays),len(arrays))
        values.append(float(np.mean([v for i in sample for v in arrays[i]])))
    return {'difference_b_minus_a':float(np.mean([v for ar in arrays for v in ar])),
            'image_cluster_bootstrap_95ci':np.quantile(values,[.025,.975]).tolist(),
            'replicates':replicates}

