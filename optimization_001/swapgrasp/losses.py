"""GT is used here during training only. Stable under AMP."""
import math
import torch
import torch.nn.functional as F


def region_response(z, support, temperature=.25):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    values = z.float()[support.bool()]
    if values.numel() == 0:
        raise ValueError('Cannot pool an empty support')
    return temperature*(torch.logsumexp(values/temperature, dim=0)-math.log(values.numel()))


def swap_loss(logits, supports, pairs, margin=.2, temperature=.25):
    terms = []
    for a,b in pairs:
        pa,pb = supports[a].bool(),supports[b].bool()
        if not pa.any() or not pb.any() or (pa & pb).any():
            continue
        da = region_response(logits[a],pa,temperature)-region_response(logits[a],pb,temperature)
        db = region_response(logits[b],pb,temperature)-region_response(logits[b],pa,temperature)
        terms.append(.5*(F.softplus(margin-da)+F.softplus(margin-db)))
    return torch.stack(terms).mean() if terms else logits.float().sum()*0


def hard_swap_loss(logits, supports, pairs, margin=.5, temperature=.25, top_fraction=.25):
    """Rank each target above the highest-scoring competing grasp supports.

    Hinge terms already satisfying the margin get zero weight. Average over all
    verified directions rather than amplifying the last few active pairs.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError('top_fraction must lie in (0,1]')
    gaps = []
    for a,b in pairs:
        pa,pb = supports[a].bool(),supports[b].bool()
        if not pa.any() or not pb.any() or (pa & pb).any():
            continue
        for query,own,other in ((a,pa,pb),(b,pb,pa)):
            negative = logits[query].float()[other]
            k = max(1,math.ceil(negative.numel()*top_fraction))
            values = negative.topk(k).values
            competing = temperature*(torch.logsumexp(values/temperature,0)-math.log(k))
            gaps.append(region_response(logits[query],own,temperature)-competing)
    if not gaps:
        zero = logits.float().sum()*0
        return zero, {'active_fraction':zero.detach(),'mean_gap':zero.detach()}
    gap = torch.stack(gaps)
    return F.relu(margin-gap).mean(), {'active_fraction':(gap < margin).float().mean().detach(),
                                    'mean_gap':gap.mean().detach()}


def swap_weight(step, loss_config):
    maximum = loss_config['lambda_swap']
    start = loss_config.get('swap_start',0)
    ramp = loss_config.get('swap_ramp',0)
    if step <= start:
        return 0.
    return maximum*min(1.,(step-start)/max(1,ramp))


def geometry_loss(prediction, target):
    """Minimum over complete valid vectors, never per-channel alternatives."""
    if target['geom_count'] == 0:
        return prediction.sum()*0
    vectors = target['geom_vectors'].to(prediction.device)
    index = target['geom_index'].to(prediction.device)
    group = target['geom_group'].to(prediction.device)
    predicted = prediction.flatten(1).T[index]
    error = F.smooth_l1_loss(predicted.float(),vectors,reduction='none',beta=.1)
    error = error[:,:2].mean(1)+error[:,2:].mean(1)
    minimum = error.new_full((target['geom_count'],), float('inf'))
    minimum = minimum.scatter_reduce(0,group,error,reduce='amin',include_self=True)
    return minimum.mean()


def grasp_loss(output, targets, lambda_geometry=1.):
    qualities, geometries = [], []
    for prediction,target in zip(output,targets):
        z = prediction[0].float()
        y,valid = target['quality'].to(z.device),target['valid'].to(z.device)
        p = z.sigmoid()
        positive = (y == 1) & valid
        negative = (y < 1) & valid
        q = (-(1-p).pow(2)*F.logsigmoid(z)*positive
             -(1-y).pow(4)*p.pow(2)*F.logsigmoid(-z)*negative).sum()/positive.sum().clamp_min(1)
        qualities.append(q)
        geometries.append(geometry_loss(prediction[1:],target))
    q,g = torch.stack(qualities).mean(),torch.stack(geometries).mean()
    return q+lambda_geometry*g, {'quality': q, 'geometry': g}

