"""Refresh all seed curves and a compact progress index while training continues."""
import json
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

runroot=Path(sys.argv[1]).resolve();protocol=json.loads((runroot/'protocol.json').read_text())
styles={'baseline':'#0072B2','swap':'#D55E00'};fig,axs=plt.subplots(1,3,figsize=(17,4))
rows=[]
for name,item in protocol['runs'].items():
    directory=Path(item['directory']);log=directory/'metrics.jsonl'
    if not log.exists():item['status']='starting';continue
    lines=[json.loads(s) for s in log.read_text().splitlines() if s.strip()]
    train=[r for r in lines if 'loss' in r];val=[r for r in lines if 'val/grasp_at1' in r]
    if train:
        axs[0].plot([r['step'] for r in train],[r['loss'] for r in train],
                    color=styles[item['method']],alpha=.38,label=name+' total')
        if item['method']=='swap':
            axs[1].plot([r['step'] for r in train],[r['quality'] for r in train],color='#009E73',alpha=.35)
            axs[1].plot([r['step'] for r in train],[r['geometry'] for r in train],color='#E69F00',alpha=.35)
            axs[1].plot([r['step'] for r in train],[r['weighted_swap'] for r in train],color='#CC79A7',alpha=.55)
        item['latest_train']=train[-1];item['latest_validation']=val[-1] if val else None
    pid=item.get('pid');path=Path('/proc')/str(pid)/'stat'
    running=path.exists() and path.read_text().split()[2]!='Z'
    item['status']='running' if running else 'finished'
    rows.append((name,item))
for ax in axs[:2]:ax.set_xlabel('optimizer step');ax.grid(alpha=.18)
axs[0].set_title('Total training loss');axs[0].set_ylabel('loss (log scale)');axs[0].set_yscale('log')
axs[1].set_title('Swap runs: quality / geometry / weighted L-swap');axs[1].set_ylabel('loss (log scale)');axs[1].set_yscale('log')
from matplotlib.lines import Line2D
axs[1].legend(handles=[Line2D([0],[0],color=c,label=label) for c,label in
                       [('#009E73','quality'),('#E69F00','geometry'),('#CC79A7','weighted L-swap')]])
for ax in axs[:2]:
    if ax is axs[0]:ax.legend(ncol=2,fontsize=7)
axs[2].set_title('Run status')
axs[2].axis('off')
status='\n'.join(f"{name:14} GPU {x['gpu']} {x['status']:9} step {(x.get('latest_train') or {}).get('step','-')} val@1 {((x.get('latest_validation') or {}).get('val/grasp_at1',0)):.3f}" for name,x in rows)
axs[2].text(0,1,status,va='top',family='monospace',fontsize=8)
fig.tight_layout();fig.savefig(runroot/'live_training.png',dpi=160);plt.close(fig)
(runroot/'live_status.json').write_text(json.dumps(protocol,indent=2))
