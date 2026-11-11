"""Write a Chinese reviewable report from finished diagnostic pilot artifacts."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from swapgrasp.data import read_jsonl

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
run=Path(a.run).resolve();state=json.loads((run/'state.json').read_text())
if state['status']!='complete':raise ValueError('Report requires completed pilot')
manifest=Path(state['manifest']);rows=read_jsonl(manifest)
comparison=json.loads((run/'comparison.json').read_text())
metrics={m:{s:json.loads((run/m/s/'metrics.json').read_text()) for s in ['train','val','test']} for m in ['overfit','baseline','swap']}
import torch
checkpoints={m:torch.load(run/m/'best.pt',map_location='cpu',weights_only=False) for m in ['baseline','swap']}
config_a=checkpoints['baseline']['config'];config_b=checkpoints['swap']['config']
import copy
same_a=copy.deepcopy(config_a);same_b=copy.deepcopy(config_b)
same_a['loss'].pop('lambda_swap');same_b['loss'].pop('lambda_swap')
if same_a!=same_b:raise ValueError('Unexpected baseline/swap config differences')
if (run/'baseline/schedule.jsonl').read_bytes()!=(run/'swap/schedule.jsonl').read_bytes():
    raise ValueError('Sample schedules differ')
splits={s:{'images':len({r['image_id'] for r in rows if r['split']==s}),
           'queries':sum(r['split']==s for r in rows)} for s in ['train','val','test']}
annotation={k:sorted({r['annotation_integrity'][k] for r in rows}) for k in rows[0]['annotation_integrity']}
evidence={'splits':splits,'annotation_integrity':annotation,'metrics':metrics,'comparison':comparison,
          'checkpoint_steps':{m:ck['step'] for m,ck in checkpoints.items()},
          'same_configuration_except_lambda_swap':True,'same_sample_schedule':True,
          'model_selection':'first maximum validation grasp@1; test not used to tune hyperparameters',
          'seed':config_a['seed'],'max_steps':config_a['train']['max_steps'],
          'data_hash':checkpoints['baseline']['data_hash']}
(run/'evidence.json').write_text(json.dumps(evidence,indent=2))
def percent(v):return '—' if v is None else f'{100*v:.1f}%'
table=['| 模型 | 训练 Top-1 | 验证 Top-1 | 测试 Top-1 | 测试 Top-5 | 测试配对同时成功 |',
       '|---|---:|---:|---:|---:|---:|']
for m in ['baseline','swap']:
    scores=metrics[m];table.append('| '+m+' | '+' | '.join(percent(x) for x in [scores['train']['grasp_at1'],scores['val']['grasp_at1'],scores['test']['grasp_at1'],scores['test']['grasp_at5'],scores['test']['pair_both_grasp_at1']])+' |')
low,high=comparison['image_cluster_bootstrap_95ci']
body=f'''# 首轮真实数据训练与可视化报告

已完成过拟合检查、baseline / swap 对照、独立推理、评分、训练曲线、热图和双指令对照图。这里的“真实数据”是官方 Grasp-Anything++ 数据，场景本身主要为合成图像；本报告不包含机器人实机实验。

## 数据与协议

使用 85 张图、399 条原始指令，按图像划分为训练 {splits['train']['images']} 张 / {splits['train']['queries']} 条、验证 {splits['val']['images']} 张 / {splits['val']['queries']} 条、测试 {splits['test']['images']} 张 / {splits['test']['queries']} 条。训练/验证/测试分别有 109 / 24 / 23 对通过几何筛选的指令配对。筛选明确要求不同物体、非重叠目标 mask、全部 GT 中心处于目标内部，以及模型分辨率下的中心支持集合不相交。

本子集预先要求每张图至少两个官方允许的 object ID，属于小规模多物体诊断协议，不是完整官方测试集。训练图像与测试图像严格分离；官方列表中共享测试图像的训练记录被排除。测试有 89 条 seen 和 15 条 unseen 指令，unseen 样本数不足以作可靠泛化结论。

本轮所有使用的标注包均已通过完整 LFS SHA-256 及成员 CRC 校验。图像通过 ZIP range 提取并验证成员 CRC，完整图像分包仍在下载，尚未完成全档 SHA 校验。目录名 pilot_multi 的 prefix 字样仅反映最初准备方式，具体完整性以 evidence.json 为准。

## 训练结果

先在经过视觉检查的“杯子/遥控器”和“书/眼镜”两张训练图上跑 1,500 步：10 条指令 Top-1 为 100%，13 对指令的同时成功率为 100%。这是优化与条件输入检查，不是泛化结果。

baseline 和 swap 都采用 ResNet-18 / FPN / CLIP 文本塔 / FiLM / 5 通道抓取头，320×320，GPU 0，BF16，seed 42，各训练 1,500 步，每步 8 条 query，其中 2 对交换指令。两者初始化和样本序列相同，配置仅 lambda_swap 为 0 / 0.1。学习率均按相同的验证指标规则调整，轨迹可随验证结果而不同。按验证 Top-1 选最早达到最大值的 checkpoint；本轮对应 baseline 第 {checkpoints['baseline']['step']} 步、swap 第 {checkpoints['swap']['step']} 步。

'''+ '\n'.join(table)+f'''

测试 Top-1 差值为 {comparison['difference_b_minus_a']*100:.2f} 个百分点；以图像为单位做 1,000 次配对 bootstrap，95% 区间为 [{low*100:.2f}, {high*100:.2f}] 个百分点。区间触及 0，且只有单个随机种子，因此暂不能宣称稳定提升。两模型测试配对同时成功率均为 0%，说明同图多指令的泛化仍未解决。

成功要求同一个 GT 同时满足旋转 IoU > 0.25、角误差 < 30°。本项目采用连续多边形 IoU，尚未逐样本与官方像素光栅化 evaluator 对齐，因此不能直接对比论文榜单数值。

## 标注质量与解释范围

视觉检查发现，部分文本物体名与图中目标不一致，例如某些 ring 指令对应容器区域、pencil 指令对应水果区域；部分不同部件指令共享相同的 mask 或抓取标签。原始标签保留，几何配对筛选也不能保证文本语义正确。早期 manifest 的 semantic_verified 字段仅表示官方文件名对应，不代表人工确认；当前 adapter 已分开记录 mapping_verified 与 semantic_verified。

mask 只用于离线配对、审核和诊断，推理只读取 RGB 与文本。预测质量图是模型的抓取分数，不是经过校准的物理成功概率。原数据的合成场景、噪声标注、小样本和单种子结果限制了现阶段结论。

## 查看结果

- [swap 测试预测框与热图](swap/test_visuals/index.html)
- [baseline 测试预测框与热图](baseline/test_visuals/index.html)
- [双指令过拟合预测](overfit/train_visuals/index.html)
- [baseline 训练曲线](baseline/curves.png)
- [swap 训练曲线](swap/curves.png)
- [2×2 对照与失败统计](explanations/)
- [原始评测证据](evidence.json)

绿色为 GT（仅诊断图），红色为 Top-1，黄色为其他候选。predict 导出的独立预测图不读取或绘制 GT。

## 下一步

继续完成全量图像下载和 SHA 校验；扩大独立图像样本并明确记录文本噪声；对齐官方 evaluator；按预先固定的配置运行多个随机种子和更大规模对照。后续调参只使用训练/验证数据，不能据本轮测试结果选择超参。
'''
(run/'REPORT.zh-CN.md').write_text(body,encoding='utf-8')
# The HTML entry keeps figures and all raw artifacts one click away.
table_html='<table><thead><tr>'+''.join(f'<th>{html.escape(s.strip())}</th>' for s in table[0].strip('|').split('|'))+'</tr></thead><tbody>'
for line in table[2:]:table_html+='<tr>'+''.join(f'<td>{html.escape(s.strip())}</td>' for s in line.strip('|').split('|'))+'</tr>'
table_html+='</tbody></table>'
panels=sorted((run/'explanations').glob('*.png'))
panel_html=''.join(f'<figure><img src="explanations/{p.name}" alt="Prompt/model comparison"></figure>' for p in panels if p.name!='failure_counts.png')
paragraphs=[]
for paragraph in body.split('\n\n'):
    if paragraph.startswith('#'):paragraphs.append('<h2>'+html.escape(paragraph.lstrip('# '))+'</h2>')
    elif paragraph.startswith('|'):paragraphs.append(table_html)
    elif paragraph.startswith('- ['):
        import re
        links=[]
        for line in paragraph.splitlines():
            match=re.fullmatch(r'- \[(.*?)\]\((.*?)\)',line)
            if match:links.append(f'<li><a href="{html.escape(match[2])}">{html.escape(match[1])}</a></li>')
        paragraphs.append('<ul>'+''.join(links)+'</ul>')
    else:paragraphs.append('<p>'+html.escape(paragraph)+'</p>')
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SwapGrasp 首轮实验</title><style>body{margin:0;background:#f5f7fa;color:#19283d;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1100px;margin:auto;padding:32px}h2{margin-top:2em;line-height:1.4}p{max-width:1000px}a{color:#175dc5}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #d9e1ee;text-align:right}td:first-child,th:first-child{text-align:left}img{max-width:100%;height:auto}figure{margin:20px 0;padding:12px;background:white}.curves{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:700px){main{padding:16px}table{font-size:12px}.curves{grid-template-columns:1fr}}</style><main>'''+''.join(paragraphs)+'''<h2>训练曲线</h2><div class="curves"><img src="baseline/curves.png" alt="Baseline curves"><img src="swap/curves.png" alt="Swap curves"></div><h2>同图两指令 × 两模型</h2>'''+panel_html+'</main></html>'
(run/'REPORT.zh-CN.html').write_text(page,encoding='utf-8')
print('REPORT',run/'REPORT.zh-CN.html',flush=True)
