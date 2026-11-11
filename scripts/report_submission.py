"""Build a Chinese results report from completed multi-seed prediction artifacts."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from swapgrasp.data import read_jsonl
from swapgrasp.geometry import grasp_success


def percentage(x):return '—' if x is None else f'{x*100:.2f}%'


def sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def prompt_specificity(run,protocol):
    rows={r['sample_id']:r for r in read_jsonl(protocol['manifest']) if r['split']=='test'}
    pairs=[p for p in read_jsonl(protocol['pairs']) if p['split']=='test']
    if not pairs:
        raise ValueError('Test split has no validated prompt pairs')
    run_metrics={};image_observations={}
    for method in ('baseline','swap'):
        for seed in protocol['seeds']:
            name=f'{method}_s{seed}'
            predictions={p['sample_id']:p['predictions'] for p in read_jsonl(run/name/'test_predictions.jsonl')}
            if not set(rows).issubset(predictions):
                raise ValueError(f'Missing predictions for prompt specificity: {name}')
            values={k:[] for k in ('target_at1','cross_at1','gap_at1','target_at5','cross_at5','gap_at5')}
            for pair in pairs:
                a,b=pair['sample_a'],pair['sample_b']
                if a not in rows or b not in rows:continue
                bucket=image_observations.setdefault((seed,pair['image_id']),{})
                for source,target in ((a,b),(b,a)):
                    ps=predictions[source]
                    own=grasp_success(ps,rows[source]['grasps'],k=1)
                    cross=grasp_success(ps,rows[target]['grasps'],k=1)
                    own5=grasp_success(ps,rows[source]['grasps'],k=5)
                    cross5=grasp_success(ps,rows[target]['grasps'],k=5)
                    observation={'target_at1':int(own),'cross_at1':int(cross),'gap_at1':int(own)-int(cross),
                                 'target_at5':int(own5),'cross_at5':int(cross5),'gap_at5':int(own5)-int(cross5)}
                    for key,value in observation.items():
                        values[key].append(value)
                        bucket.setdefault((method,key),[]).append(value)
            run_metrics[name]={'directions':len(values['target_at1']),'pairs':len(pairs),
                               'pair_images':len({p['image_id'] for p in pairs}),
                               **{k:float(np.mean(v)) if v else None for k,v in values.items()}}
    summary={}
    for method in ('baseline','swap'):
        summary[method]={}
        for metric in ('target_at1','cross_at1','gap_at1','target_at5','cross_at5','gap_at5'):
            vals=[run_metrics[f'{method}_s{seed}'][metric] for seed in protocol['seeds']]
            summary[method][metric]={'n':len(vals),'mean':float(np.mean(vals)),
                                     'sample_std':float(np.std(vals,ddof=1)) if len(vals)>1 else None}
    comparisons={}
    for seed in protocol['seeds']:
        image_ids=sorted(i for s,i in image_observations if s==seed)
        rng=np.random.default_rng(9000+seed);boot={k:[] for k in ('target_at1','cross_at1','gap_at1')}
        for _ in range(1000):
            sample=rng.choice(image_ids,len(image_ids),replace=True)
            for metric in boot:
                difference=[]
                for method in ('baseline','swap'):
                    vals=[v for image in sample for v in image_observations[seed,image][method,metric]]
                    difference.append(float(np.mean(vals)))
                boot[metric].append(difference[1]-difference[0])
        comparisons[str(seed)]={}
        for metric,replicates in boot.items():
            by_method=[]
            for method in ('baseline','swap'):
                vals=[v for image in image_ids for v in image_observations[seed,image][method,metric]]
                by_method.append(float(np.mean(vals)))
            comparisons[str(seed)][metric]={'swap_minus_baseline':by_method[1]-by_method[0],
                                            'image_cluster_bootstrap_95ci':np.quantile(replicates,[.025,.975]).tolist(),
                                            'replicates':1000,'images':len(image_ids)}
    return {'definition':'For each validated same-image object pair, compare each existing prompt-conditioned prediction with its own target and the competing target. This measures cross-target overlap; it does not rerun inference with a replaced prompt.',
            'runs':run_metrics,'summary':summary,'seed_comparisons':comparisons}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    run=Path(a.run).resolve();protocol=json.loads((run/'protocol.json').read_text())
    metrics={};details={};checks={};steps={}
    for name,item in protocol['runs'].items():
        folder=Path(item['directory'])
        if not (folder/'test/metrics.json').exists():continue
        metrics[name]=json.loads((folder/'test/metrics.json').read_text())
        details[name]=json.loads((folder/'test/detailed_metrics.json').read_text())
        checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        steps[name]=checkpoint['step']
    comparisons=json.loads((run/'comparisons.json').read_text())
    for seed in protocol['seeds']:
        ba,sw=(run/f'{method}_s{seed}' for method in ('baseline','swap'))
        checks[str(seed)]=((ba/'schedule.jsonl').read_bytes()==(sw/'schedule.jsonl').read_bytes())
    summary={}
    for method in ('baseline','swap'):
        names=[f'{method}_s{s}' for s in protocol['seeds'] if f'{method}_s{s}' in metrics]
        summary[method]={}
        for metric in ('grasp_at1','grasp_at5','grasp_at1_image_macro','pair_both_grasp_at1','seen_grasp_at1','unseen_grasp_at1'):
            values=[metrics[name][metric] for name in names if metrics[name][metric] is not None]
            summary[method][metric]={'n':len(values),'mean':float(np.mean(values)) if values else None,
                                    'sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None}
        for threshold in ('strict','very_strict'):
            values=[details[name]['thresholds'][threshold]['at1'] for name in names]
            summary[method][threshold+'_at1']={'n':len(values),'mean':float(np.mean(values)) if values else None,
                                             'sample_std':float(np.std(values,ddof=1)) if len(values)>1 else None}
    matched_seeds=[seed for seed in protocol['seeds']
                   if all(f'{method}_s{seed}' in metrics for method in ('baseline','swap'))]
    specificity=prompt_specificity(run,{**protocol,'seeds':matched_seeds}) if matched_seeds else {'runs':{},'seed_comparisons':{}}
    evidence={'protocol':protocol,'test_metrics':metrics,'detailed_metrics':details,'summary':summary,
              'prompt_specificity':specificity,
              'paired_comparisons':comparisons,'same_schedule':checks,'selected_steps':steps,
              'schedule_sha256':{name:sha256_file(Path(item['directory'])/'schedule.jsonl') for name,item in protocol['runs'].items()},
              'selected_checkpoint_sha256':{name:sha256_file(Path(protocol['runs'][name]['directory'])/'best.pt') for name in metrics},
              'test_predictions_sha256':{name:sha256_file(Path(protocol['runs'][name]['directory'])/'test_predictions.jsonl') for name in metrics},
              'manifest_sha256':hashlib.sha256(Path(protocol['manifest']).read_bytes()).hexdigest(),
              'metric_notes':{'standard':'same target GT IoU>0.25 AND angle<30 degrees',
                              'strict':'same target GT IoU>0.50 AND angle<15 degrees',
                              'very_strict':'same target GT IoU>0.75 AND angle<10 degrees',
                              'geometry_reference':'One highest-IoU GT per top-1 prediction; nearest centre breaks ties.',
                              'std':'Sample standard deviation across seeds; not a confidence interval.',
                              'iou':'continuous polygon IoU, not yet aligned with official rasterization evaluator'}}
    (run/'evidence.json').write_text(json.dumps(evidence,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'baseline':'#0072B2','swap':'#D55E00'}
    fig,axs=plt.subplots(1,2,figsize=(10,3.5))
    for method in ('baseline','swap'):
        names=[f'{method}_s{s}' for s in protocol['seeds'] if f'{method}_s{s}' in details]
        for idx,k in enumerate((1,5)):
            xs=[.25,.35,.50,.65,.75];means=[];std=[]
            for x in xs:
                vals=[next(t[f'at{k}'] for t in details[name]['threshold_grid'] if t['iou_gt']==x and t['angle_lt_degrees']==15) for name in names]
                means.append(100*np.mean(vals));std.append(100*np.std(vals,ddof=1) if len(vals)>1 else 0.)
            axs[idx].plot(xs,means,marker='o',label=method,color=colors[method])
            axs[idx].fill_between(xs,np.array(means)-std,np.array(means)+std,color=colors[method],alpha=.12)
            axs[idx].set(title=f'Top-{k}, angle error < 15 degrees',xlabel='Rotated IoU threshold',ylabel='Success (%)',ylim=(0,100))
            axs[idx].grid(alpha=.18);axs[idx].legend()
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(run/f'threshold_curves.{suffix}',dpi=200)
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.4))
    for method in ('baseline','swap'):
        path=run/f'{method}_s42/test/detailed_per_query.jsonl'
        if not path.exists():continue
        data=read_jsonl(path)
        for ax,key,label in zip(axs,('iou','center_px','angle_degrees'),('Top-1 best-reference IoU','Centre error (pixels)','Angle error (degrees)')):
            values=sorted(r['top1_geometry'][key] for r in data if r['top1_geometry'] is not None)
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=method,color=colors[method])
            ax.set(xlabel=label,ylabel='Fraction of queries',ylim=(0,1));ax.grid(alpha=.18);ax.legend()
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(run/f'geometry_cdf.{suffix}',dpi=200)
    plt.close(fig)
    table=['| 模型 / seed | Top-1 | Top-5 | 严格 Top-1 | 更严格 Top-1 | 成对 Top-1 | 最佳 step |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():
        d=details[name]['thresholds'];table.append('| '+name+' | '+' | '.join(percentage(x) for x in
             (m['grasp_at1'],m['grasp_at5'],d['strict']['at1'],d['very_strict']['at1'],m['pair_both_grasp_at1']))+f' | {steps[name]} |')
    means=['| 模型 | 标准 Top-1 | 标准 Top-5 | 严格 Top-1 | 更严格 Top-1 | 成对 Top-1 |',
           '|---|---:|---:|---:|---:|---:|']
    for method in ('baseline','swap'):
        values=[]
        for key in ('grasp_at1','grasp_at5','strict_at1','very_strict_at1','pair_both_grasp_at1'):
            m=summary[method][key]
            values.append(percentage(m['mean'])+(f" ± {100*m['sample_std']:.2f}" if m['sample_std'] is not None else ''))
        means.append('| '+method+' | '+' | '.join(values)+' |')
    diffs=[]
    for seed,cmp in comparisons.items():
        c=cmp['grasp_at1'];lo,hi=c['image_cluster_bootstrap_95ci']
        diffs.append(f"seed {seed}：Swap−Baseline={c['difference_b_minus_a']*100:.2f} 个百分点，图像聚类配对 bootstrap 95% 区间 [{lo*100:.2f}, {hi*100:.2f}]。")
    interpretation='六组结果未齐，暂不解读三种子均值差。'
    if len(metrics)==len(protocol['runs']) and len(comparisons)==len(protocol['seeds']):
        names={'grasp_at1':'标准 Top-1','grasp_at5':'Top-5','strict_at1':'严格 Top-1','pair_both_grasp_at1':'成对 Top-1'}
        deltas='、'.join(f"{label} {100*(summary['swap'][key]['mean']-summary['baseline'][key]['mean']):+.2f}"
                        for key,label in names.items() if all(summary[m][key]['mean'] is not None for m in ('baseline','swap')))
        positive=sum(c['grasp_at1']['difference_b_minus_a']>0 for c in comparisons.values())
        supported=sum(c['grasp_at1']['image_cluster_bootstrap_95ci'][0]>0 for c in comparisons.values())
        interpretation=(f"SwapGrasp 相对 Baseline 的均值变化：{deltas} 个百分点。"
                        f"{len(comparisons)} 个种子中，{positive} 个 Top-1 差值为正，"
                        f"{supported} 个图像聚类 bootstrap 区间整体高于零。")
        interpretation+=('当前结果支持这个子集上跨种子一致的正向 Top-1 差值。' if supported==len(comparisons) and supported
                         else '当前结果未建立跨种子一致的正向 Top-1 收益。')
    supplemental=['| 模型 | 图像宏平均 Top-1 | Seen Top-1 | Unseen Top-1 |',
                  '|---|---:|---:|---:|']
    for method in ('baseline','swap'):
        values=[]
        for key in ('grasp_at1_image_macro','seen_grasp_at1','unseen_grasp_at1'):
            item=summary[method][key]
            values.append(percentage(item['mean'])+(f" ± {100*item['sample_std']:.2f}" if item['sample_std'] is not None else ''))
        supplemental.append('| '+method+' | '+' | '.join(values)+' |')
    body='''# SwapGrasp 提交版训练与评测报告

这是六组实际训练与独立测试产生的结果，原始证据见 evidence.json。模型输入为 RGB 与文本；目标 mask 不进入推理。三种子均值与标准差用于描述运行间波动，不能替代图像级配对置信区间。未作机器人实机实验。

## 数据与训练

'''+f"使用 {protocol['images']} 张图像、{protocol['queries']} 条指令；图像划分 {protocol['images_by_split']}，指令划分 {protocol['queries_by_split']}。同图配对数量 {protocol['pairs_by_split']}。完整归档通过 SHA-256 校验，提取成员通过 CRC；362 条可疑亚像素几何记录被固定规则隔离。\n\n"+f"每组使用 A100 80GB、BF16、batch {protocol['batch_size_queries']}，包含 {protocol['pairs_per_step']} 对指令，训练 {protocol['max_steps']} 步。训练采样有一半 query 来自配对，约 {protocol['epochs_equivalent']} 个等效 query epoch 不是均匀遍历整个训练集。三个种子为 {protocol['seeds']}；同种子两模型的初始化、样本顺序和配置相同，仅 lambda_swap=0/0.1。学习率按相同验证规则调节，实际轨迹可不同。最佳 checkpoint 只按验证 Top-1 选择。\n\n"+'''## 三种子汇总

'''+ '\n'.join(means)+'''\n\n表中 ± 是种子间样本标准差（百分点），不是 95% 置信区间。

## 每次运行

'''+ '\n'.join(table)+'\n\n'+'\n\n'.join(diffs)+'''\n\n## 指标解释

标准：同一目标的同一个 GT 同时满足 IoU > 0.25、角差 < 30°。严格：IoU > 0.50、角差 < 15°。更严格：IoU > 0.75、角差 < 10°。Top-1 检查第一候选；Top-5 检查前五个中是否有合格框，是候选集合的 oracle 指标。成对成功要求一张图的两条不同目标指令都成功。阈值、候选解码和 checkpoint 不使用测试分数调节。

几何误差使用 Top-1 与一个最高 IoU 的目标 GT 匹配，再对同一个 GT 计算中心、角度和宽高误差；IoU 相同时按最近中心破平。所有有预测的 query 均纳入，不能将各分量各自选最优 GT。中心误差单位为原始图像像素，图像尺寸可能不同；这里报告的是数据集几何对齐，不是物理夹爪误差。

## 查看图表

- [训练总览与 loss](live_training.png)
- [多阈值曲线](threshold_curves.png)
- [几何误差分布](geometry_cdf.png)
- [Swap 测试预测框与热图（固定的 48 条示例）](swap_s42/test_visuals/index.html)
- [Baseline 测试预测框与热图（相同示例）](baseline_s42/test_visuals/index.html)
- [同图双指令与两模型对照](#prompt-pairs)
- [单张图片与指令推理演示](inference_demo/index.html)
- [原始完整评测证据](evidence.json)

## 解释边界

这批数据来自官方 object split 所覆盖且实际有正标注的图像；排除了与测试共享的训练图像，又做图像级验证划分，属于自定义诊断协议。不能当作完整百万图像官方 benchmark，也不能直接与论文榜单横向比较。本实现使用连续旋转多边形 IoU，尚未与官方光栅化 evaluator 逐样本对齐。原数据主要为合成场景，部分指令、物体名与部件标注存在语义噪声。几何审计不等同人工语义确认。报告如实保留所有种子，无论是否支持预期提升。
'''
    reading='## 先读结论\n\n'+interpretation+'\n\n'
    reading+='标准指标看能否提出大体合理的目标抓取；严格指标看几何精度；成对指标看同一图像换目标后是否两条指令都成功。Top-5 不能替代 Top-1，也不等于真实机器人成功率。只有 36 张测试图像提供有效同图配对，配对和特异性诊断的泛化证据有限。\n\n'
    reading+='## 图像与类别分项\n\n'+'\n'.join(supplemental)+'\n\n'
    reading+='图像宏平均先在每张图上算指令成功率，再对图像等权平均，避免指令较多的图像主导结果。Seen/Unseen 来自原始官方类别清单；本报告的图像隔离与标注过滤使其成为自定义子集分项，不能与官方全量数值直接排名。\n\n'
    body=body.replace('## 三种子汇总',reading+'## 三种子汇总')
    specificity_table=['| 模型 / seed | 正确目标 Top-1 | 命中竞争目标 Top-1 | 目标特异性差值 | 正确目标 Top-5 | 竞争目标 Top-5 |',
                       '|---|---:|---:|---:|---:|---:|']
    for name,item in specificity['runs'].items():
        specificity_table.append('| '+name+' | '+' | '.join(percentage(item[key]) for key in
            ('target_at1','cross_at1','gap_at1','target_at5','cross_at5'))+' |')
    specifics='## 同图目标特异性\n\n这里将已生成的某条指令预测与同图另一目标的 GT 比较，并非重新输入错误指令。差值等于正确目标命中率减去竞争目标命中率，单位为百分点。每个指令可能随不同配对重复计数，因而表中正确目标率的统计总体不同于全测试集。此为训练启动后增加的补充诊断；不用于选模型或调参。\n\n'
    specifics+='\n'.join(specificity_table)+'\n\n'
    for seed,cmp in specificity['seed_comparisons'].items():
        gap=cmp['gap_at1'];lo,hi=gap['image_cluster_bootstrap_95ci']
        specifics+=f"seed {seed}：Swap−Baseline 的特异性差值为 {gap['swap_minus_baseline']*100:+.2f} 个百分点，图像聚类 bootstrap 95% 区间 [{lo*100:+.2f}, {hi*100:+.2f}]。\n\n"
    body=body.replace('## 每次运行',specifics+'## 每次运行')
    if len(metrics)!=len(protocol['runs']):
        body=body.replace('这是六组实际训练与独立测试产生的结果',f'这是已完成独立测试的 {len(metrics)}/{len(protocol["runs"])} 组结果，汇总为阶段性数据')
    (run/'REPORT.zh-CN.md').write_text(body,encoding='utf-8')
    chunks=[]
    for block in body.split('\n\n'):
        if block.startswith('#'):
            level=min(3,len(block)-len(block.lstrip('#')));chunks.append(f'<h{level}>'+html.escape(block.lstrip('# '))+f'</h{level}>')
        elif block.startswith('|'):
            lines=block.splitlines();part='<table><thead><tr>'+''.join('<th>'+html.escape(x.strip())+'</th>' for x in lines[0].strip('|').split('|'))+'</tr></thead><tbody>'
            for line in lines[2:]:part+='<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>'
            chunks.append(part+'</tbody></table>')
        elif block.startswith('- ['):
            items=[]
            for line in block.splitlines():
                m=re.fullmatch(r'- \[(.*?)\]\((.*?)\)',line)
                if m:items.append('<li><a href="'+html.escape(m[2])+'">'+html.escape(m[1])+'</a></li>')
            chunks.append('<ul>'+''.join(items)+'</ul>')
        else:chunks.append('<p>'+html.escape(block)+'</p>')
    figures=''.join(f'<figure><img src="{name}"></figure>' for name in ('live_training.png','threshold_curves.png','geometry_cdf.png'))
    for seed in protocol['seeds']:
        figures+=f'<h2>Seed {seed} loss 与验证曲线</h2><div class="grid">'+''.join(f'<figure><figcaption>{m}</figcaption><img src="{m}_s{seed}/curves.png"></figure>' for m in ('baseline','swap'))+'</div>'
    panels=''.join(f'<figure><img src="explanations/{p.name}"></figure>' for p in sorted((run/'explanations').glob('*.png')) if p.name!='failure_counts.png')
    if (run/'explanations/failure_counts.png').exists():
        panels+='<h2>全部测试指令的失败分布（SwapGrasp，seed 42）</h2><p>这是按现有标注自动归类的诊断：success 表示命中目标 GT；matches_other_annotated_object 表示未命中目标但命中同图其他已标注物体；unresolved_localization_or_geometry 包括其余定位或几何误差。类别不等于实机执行结果。</p><figure><img src="explanations/failure_counts.png"></figure>'
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SwapGrasp 提交版结果</title><style>body{font:16px/1.85 system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fa;color:#17253a;margin:0}main{max-width:1150px;margin:auto;padding:32px}h1,h2,h3{line-height:1.4}table{border-collapse:collapse;background:white;width:100%;font-size:14px}th,td{border-bottom:1px solid #dae2ed;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%}a{color:#175dc5}figure{margin:18px 0;padding:12px;background:white}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:750px){main{padding:12px}.grid{display:block}table{font-size:11px}}</style><main>'''+''.join(chunks)+figures+'<h2 id="prompt-pairs">同图两指令 × 两模型</h2><p>绿色：全部 GT；红色：Top-1；黄色：其他候选。左列 Baseline，右列 Swap，上下两行分别为同图的不同指令；热图色标固定为 sigmoid 分数 [0,1]。固定选择前 12 个不同图像的有效配对，不按测试成败筛选。</p>'+panels+'</main></html>'
    (run/'REPORT.zh-CN.html').write_text(page,encoding='utf-8')
    print('REPORT',run/'REPORT.zh-CN.html',flush=True)


if __name__=='__main__':main()
