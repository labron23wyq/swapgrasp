"""Complete the fixed optimization comparison; test never selects models."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from swapgrasp.data import read_jsonl,write_jsonl
from swapgrasp.metrics import extended_scores,paired_bootstrap
run=root.parent/'runs/optimization_001'
python=str(root.parent/'.venv/bin/python')
protocol=json.loads((run/'protocol.json').read_text())
manifest=Path(protocol['manifest']);pairs=Path(protocol['pairs'])
state={'status':'waiting_training','errors':{},'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}


def save_state(status):
    state['status']=status
    temp=run/'workflow_state.json.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(run/'workflow_state.json')


def command(name,args,gpu):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (run/(name+'.log')).open('a') as log:
        result=subprocess.run([python,*args],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(name+' failed: '+(run/(name+'.log')).read_text()[-2000:])


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def current_progress():
    progress={}
    for name,item in protocol['runs'].items():
        folder=Path(item['directory'])
        if not (folder/'metrics.jsonl').exists():continue
        records=read_jsonl(folder/'metrics.jsonl');tr=[r for r in records if 'loss' in r];va=[r for r in records if 'val/grasp_at1' in r]
        if tr:progress[name]={'step':tr[-1]['step'],'loss':tr[-1]['loss'],'seconds':tr[-1]['seconds'],
            'latest_val_top1':va[-1]['val/grasp_at1'] if va else None,
            'best_val_top1':max(r['val/grasp_at1'] for r in va) if va else None}
    (run/'progress.json').write_text(json.dumps(progress,indent=2))
    return progress


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
    colors={'original':'#0072B2','balanced':'#009E73','hard_swap':'#D55E00'}
    for name,item in protocol['runs'].items():
        path=Path(item['directory'])/'metrics.jsonl'
        if not path.exists():continue
        rows=read_jsonl(path);tr=[r for r in rows if 'loss' in r];va=[r for r in rows if 'val/grasp_at1' in r]
        style='-' if item['seed']==42 else '--';color=colors[item['method']]
        for ax,key in ((axes[0,0],'loss'),(axes[0,1],'geometry'),(axes[1,1],'swap')):
            ax.plot([r['step'] for r in tr],[r[key] for r in tr],style,c=color,label=name)
        axes[1,0].plot([r['step'] for r in va],[100*r['val/grasp_at1'] for r in va],style,c=color,label=name)
    for ax,title in zip(axes.flat,['Total training loss','Geometry loss','Validation Top-1 (%)','Swap objectives (different definitions)']):
        ax.set_title(title);ax.set_xlabel('Step');ax.grid(alpha=.2)
    axes[0,0].set_yscale('log');axes[1,1].set_yscale('symlog',linthresh=.001)
    axes[0,0].legend(fontsize=7);fig.savefig(run/'training_curves.png',dpi=170);plt.close(fig)


def fit(seed,gpu):
    folder=run/f'balanced_s{seed}';out=run/f'reranked_s{seed}'
    command(f'rerank_fit_s{seed}',['scripts/rerank_candidates.py','--checkpoint',str(folder/'best.pt'),
          '--manifest',str(manifest),'--pairs',str(pairs),'--output',str(out),'--stage','fit'],gpu)


def evaluate_seed(seed,gpu):
    for method in ('original','balanced','hard_swap'):
        name=f'{method}_s{seed}';folder=run/name
        command(name+'_test_predict',['evaluate.py','predict','--checkpoint',str(folder/'best.pt'),
              '--manifest',str(manifest),'--split','test','--batch-size','32',
              '--output',str(folder/'test_predictions.jsonl')],gpu)
        command(name+'_test_score',['evaluate.py','score','--manifest',str(manifest),'--pairs',str(pairs),
              '--split','test','--predictions',str(folder/'test_predictions.jsonl'),'--output',str(folder/'test')],gpu)
    folder=run/f'balanced_s{seed}';out=run/f'reranked_s{seed}'
    command(f'rerank_test_s{seed}',['scripts/rerank_candidates.py','--checkpoint',str(folder/'best.pt'),
          '--manifest',str(manifest),'--pairs',str(pairs),'--output',str(out),'--stage','test'],gpu)
    # Baseline and reranker use exactly the same proposal pool.
    import torch
    cached=torch.load(out/'test_cache.pt',map_location='cpu',weights_only=False)['proposals']
    base=read_jsonl(folder/'test_predictions.jsonl')
    if cached!=base:raise ValueError('Frozen candidate extractor does not exactly reproduce baseline')


def report():
    metrics={};per_query={};seeds=protocol['seeds']
    for method in ('original','balanced','hard_swap','reranked'):
        for seed in seeds:
            name=f'{method}_s{seed}';folder=run/name
            metric=folder/('test_metrics.json' if method=='reranked' else 'test/detailed_metrics.json')
            detail=folder/('test_per_query.jsonl' if method=='reranked' else 'test/detailed_per_query.jsonl')
            metrics[name]=json.loads(metric.read_text());per_query[name]=read_jsonl(detail)
    comparisons={}
    for a,b in [('original','balanced'),('balanced','hard_swap'),('balanced','reranked'),('original','reranked')]:
        for seed in seeds:
            comparisons[f'{a}_vs_{b}_s{seed}']={key:paired_bootstrap(per_query[f'{a}_s{seed}'],per_query[f'{b}_s{seed}'],metric=key)
                 for key in ('grasp_at1','strict_at1','very_strict_at1')}
    summary={};lines=['# 优化对照实验结果','',
       '本轮使用新的图像隔离协议，从 ImageNet/CLIP 预训练权重重新训练，不能把本表数值与旧测试集直接相减。旧官方 test split 完全排除。原 train 按图像重分为新 train/新 test；新 test 从未进入本轮训练或选模，但它曾进入上一轮六模型训练，且本轮优化想法是在查看上一轮结果后提出，因此这是本轮的 held-out split，不是独立的确认性 benchmark。原 val 分为 detector/scorer 选模 val 与 scorer 拟合 rankfit；所有选择在读取本轮新 test 前固定。','',
       '| 方法 | Top-1 | Top-5 | 严格 Top-1 | 更严格 Top-1 | 成对 Top-1 |',
       '|---|---:|---:|---:|---:|---:|']
    for method in ('original','balanced','hard_swap','reranked'):
        group=[metrics[f'{method}_s{s}'] for s in seeds]
        values={'top1':[m['standard_metrics']['grasp_at1'] for m in group],
                'top5':[m['standard_metrics']['grasp_at5'] for m in group],
                'strict':[m['thresholds']['strict']['at1'] for m in group],
                'very_strict':[m['thresholds']['very_strict']['at1'] for m in group],
                'pair':[m['standard_metrics']['pair_both_grasp_at1'] for m in group]}
        summary[method]={key:{'mean':statistics.mean(v),'sample_std':statistics.stdev(v),'seeds':v}
                         for key,v in values.items() if all(x is not None for x in v)}
        text=[f"{100*summary[method][key]['mean']:.2f} ± {100*summary[method][key]['sample_std']:.2f}" if key in summary[method] else 'N/A'
              for key in values]
        lines.append('| '+method+' | '+' | '.join(text)+' |')
    lines+=['','单位为百分比；± 为两个种子的样本标准差，证据有限。strict: IoU>0.5 且角差<15°；very strict: IoU>0.75 且角差<10°。','',
            '## 配对比较','']
    for name,values in comparisons.items():
        item=values['grasp_at1'];ci=item['image_cluster_bootstrap_95ci']
        lines.append(f"- {name}: Top-1 差 {100*item['difference_b_minus_a']:+.2f} 个百分点，图像聚类 bootstrap 95% 区间 [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]。")
    lines+=['','## 重排选择','']
    for seed in seeds:
        choice=json.loads((run/f'reranked_s{seed}/choice.json').read_text())
        lines.append(f"- seed {seed}: 验证集选择 epoch={choice['epoch']}，残差权重={choice['alpha']}；权重为 0 代表验证集选择保留原排序。")
    lines+=['','## 数据与解释边界','',json.dumps(protocol['protocol'],ensure_ascii=False,indent=2),
          '','重排头只在 rankfit 图像上训练，检测器未见过这些图像；无 GT 或 mask 进入候选生成、特征抽取或推理。重排仅改变五个候选的顺序，Top-5 必须保持一致。',
          '重排头使用了 detector baseline 未使用的 351 张 rankfit 图像标签；因此对比反映额外监督加候选重排的组合效果，不能把收益全部归因于重排结构。original 对比 balanced 同时改变配对比例与图像均衡，也不是两个因素的独立消融。',
          '不同采样造成训练 batch 分布不同，不宜直接比较绝对训练 loss；hard_swap 与原 swap 的公式不同，图中的原始数值只作各自训练趋势观察。',
          '本轮新 holdout 来自旧训练图像、类别均为 seen；不是新增官方 unseen benchmark。新实验未使用旧模型权重，但研究方向已受旧数据结果启发。',
          '由于上一轮基准训练使用过这些原 train 图像，且本轮提出的优化方案参考了上一轮结果，不能将该 holdout 解释成对研究者完全盲测的独立确认。投稿级结论需要新场景或新增图像再次验证。',
          '本轮只跑两个种子、配对图像仍较少；bootstrap 区间不覆盖所有训练随机性，也不能替代真实机器人验证。','',
          '[训练曲线](training_curves.png) · [预测可视化对照](visuals/index.html) · [原始数据](evidence.json) · [冻结的选模记录](selection_frozen.json)']
    for seed in seeds:
        assert metrics[f'balanced_s{seed}']['standard_metrics']['grasp_at5']==metrics[f'reranked_s{seed}']['standard_metrics']['grasp_at5']
    audit_final(per_query)
    visual_comparisons(per_query)
    evidence={'protocol':protocol,'metrics':metrics,'summary':summary,'comparisons':comparisons,'audit':'passed',
              'candidate_sets_identical':True,'source_sha256':json.loads((root/'source_sha256.json').read_text())}
    (run/'evidence.json').write_text(json.dumps(evidence,indent=2))
    markdown='\n'.join(lines);(run/'REPORT.zh-CN.md').write_text(markdown)
    import markdown as markdown_module
    content=markdown_module.markdown(markdown,extensions=['tables','fenced_code'])
    (run/'REPORT.zh-CN.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SwapGrasp 优化对照</title><style>body{max-width:1150px;margin:40px auto;font:17px/1.7 system-ui;color:#182033;padding:0 20px}table{border-collapse:collapse;width:100%;font-size:16px}td,th{padding:10px;border-bottom:1px solid #ccd}img{max-width:100%}pre{white-space:pre-wrap;background:#f3f5f8;padding:15px}</style>'+content+'<img src="training_curves.png">')
    print(json.dumps(summary,indent=2),flush=True)


def audit_final(per_query):
    rows=read_jsonl(manifest);by_split=defaultdict(set)
    for row in rows:by_split[row['split']].add(row['image_id'])
    assert sum(map(len,by_split.values()))==len(set.union(*by_split.values()))
    ids={r['sample_id'] for r in rows if r['split']=='test'}
    checks={}
    for name,item in protocol['runs'].items():
        folder=Path(item['directory']);records=read_jsonl(folder/'metrics.jsonl')
        assert any(r['step']==protocol['max_steps'] and 'val/grasp_at1' in r for r in records)
        assert {r['sample_id'] for r in per_query[name]}==ids
        count=0
        with (folder/'schedule.jsonl').open() as stream:
            for line in stream:
                entry=json.loads(line);count+=1
                assert entry['step']==count and len(entry['sample_ids'])==256
        assert count==protocol['max_steps']
        checks[name]={'trained_steps':count,'test_predictions':len(per_query[name]),
                      'checkpoint_sha256':digest(folder/'best.pt')}
    for seed in protocol['seeds']:
        assert digest(run/f'balanced_s{seed}/schedule.jsonl')==digest(run/f'hard_swap_s{seed}/schedule.jsonl')
        assert {r['sample_id'] for r in per_query[f'reranked_s{seed}']}==ids
        base={r['sample_id']:r for r in per_query[f'balanced_s{seed}']}
        assert all(base[r['sample_id']]['grasp_at5']==r['grasp_at5'] for r in per_query[f'reranked_s{seed}'])
    (run/'audit.json').write_text(json.dumps({'passed':True,'image_splits_disjoint':True,
          'balanced_hard_schedules_identical':True,'checks':checks},indent=2))


def visual_comparisons(per_query):
    from PIL import Image,ImageDraw,ImageFont
    from swapgrasp.visualization import overlay
    rows=[r for r in read_jsonl(manifest) if r['split']=='test']
    selected=list(rows[:12]);used={r['sample_id'] for r in selected}
    base={r['sample_id']:r for r in per_query['balanced_s42']}
    new={r['sample_id']:r for r in per_query['reranked_s42']}
    for improved in (True,False):
        candidates=[r for r in rows if (new[r['sample_id']]['grasp_at1']>base[r['sample_id']]['grasp_at1'] if improved else
                    new[r['sample_id']]['grasp_at1']<base[r['sample_id']]['grasp_at1'])]
        for row in candidates[:6]:
            if row['sample_id'] not in used:selected.append(row);used.add(row['sample_id'])
    predictions={}
    for method in ('original','balanced','hard_swap','reranked'):
        predictions[method]={r['sample_id']:r['predictions'] for r in read_jsonl(run/f'{method}_s42/test_predictions.jsonl')}
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),16) if font_path.exists() else ImageFont.load_default()
    output=run/'visuals';output.mkdir(exist_ok=True);cards=[]
    for index,row in enumerate(selected):
        with Image.open(row['image']) as source:
            tiles=[]
            for method in predictions:
                panel=overlay(source,row['grasps'],predictions[method][row['sample_id']]);panel.thumbnail((500,420))
                tile=Image.new('RGB',(500,470),'white');tile.paste(panel,((500-panel.width)//2,35))
                draw=ImageDraw.Draw(tile);draw.text((12,7),method+' / seed 42',fill='black',font=font);tiles.append(tile)
        sheet=Image.new('RGB',(1000,940),'white')
        for j,tile in enumerate(tiles):sheet.paste(tile,((j%2)*500,(j//2)*470))
        name=f'comparison_{index:02d}.png';sheet.save(output/name)
        old=base[row['sample_id']]['grasp_at1'];newhit=new[row['sample_id']]['grasp_at1']
        origin='fixed manifest example' if index<12 else 'post-hoc ranking change example'
        cards.append('<article><p>'+html.escape(row['text'])+'</p><p>'+origin+f'; balanced={old}, reranked={newhit}</p><img src="'+name+'"><code>'+row['sample_id']+'</code></article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Optimization prediction comparison</title><style>body{max-width:1100px;margin:30px auto;font:17px system-ui}img{width:100%}article{margin-bottom:35px;border-bottom:1px solid #ccc}code{overflow-wrap:anywhere}</style><h1>Optimization comparison</h1><p>Green: GT. Red: Top-1. Amber: other candidates. First 12 examples fixed by manifest order; remaining examples show both beneficial and harmful reranking after evaluation.</p>'+''.join(cards))


def main():
    try:
        while True:
            progress=current_progress();plot();save_state('training')
            finished=[]
            for name,item in protocol['runs'].items():
                folder=Path(item['directory']);text=(folder/'train.log').read_text(errors='replace')
                done=f"FINISHED {protocol['max_steps']} steps;" in text
                stat=Path('/proc')/str(item['pid'])/'stat'
                alive=stat.exists() and stat.read_text().split()[2]!='Z'
                if not alive and not done:raise RuntimeError(name+' exited: '+text[-2000:])
                finished.append(done)
            if all(finished):break
            time.sleep(45)
        save_state('fitting_rerankers')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(fit,seed,6+i) for i,seed in enumerate(protocol['seeds'])]
            for future in futures:future.result()
        frozen={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'checkpoints':{},'reranker_choices':{}}
        for name,item in protocol['runs'].items():
            folder=Path(item['directory']);records=read_jsonl(folder/'metrics.jsonl')
            values=[r for r in records if 'val/grasp_at1' in r];best=max(values,key=lambda r:r['val/grasp_at1'])
            frozen['checkpoints'][name]={'sha256':digest(folder/'best.pt'),'selected_step':best['step'],'val_top1':best['val/grasp_at1']}
        for seed in protocol['seeds']:
            folder=run/f'reranked_s{seed}';frozen['reranker_choices'][str(seed)]={'choice':json.loads((folder/'choice.json').read_text()),'sha256':digest(folder/'best.pt')}
        (run/'selection_frozen.json').write_text(json.dumps(frozen,indent=2))
        save_state('evaluating_frozen_test')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(evaluate_seed,seed,6+i) for i,seed in enumerate(protocol['seeds'])]
            for future in futures:future.result()
        save_state('reporting');report();plot()
        save_state('complete')
    except Exception as error:
        state['errors']['workflow']=repr(error);save_state('failed');raise


if __name__=='__main__':main()
