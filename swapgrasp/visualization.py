import base64
import csv
import hashlib
import html
import io
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .geometry import corners,grasp_success


class PredictionExporter:
    """Save prediction artifacts from RGB/text only; never access labels."""
    def __init__(self, output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.cards=[]

    def add(self,dataset,row,predictions,quality,transform):
        key=hashlib.sha256(row['sample_id'].encode()).hexdigest()[:24]
        with Image.open(dataset.path(row['image'])) as source:
            image=source.convert('RGB')
        # Map a heatmap in letterboxed model coordinates to original image pixels.
        scale,left,top=transform['scale'],transform['left'],transform['top']
        heat=Image.fromarray(np.asarray(quality,dtype=np.float32)).transform(
            image.size,Image.Transform.AFFINE,(scale,0,left,0,scale,top),
            resample=Image.Resampling.BILINEAR)
        for suffix,values in [('prediction',None),('heatmap',np.asarray(heat))]:
            tile=overlay(image,[],predictions,quality=values)
            caption=row['text']
            if predictions:
                import math
                x,y,w,h,angle=predictions[0]['grasp']
                caption+=f' | top-1: ({x:.1f}, {y:.1f}), {w:.1f} x {h:.1f} px, {math.degrees(angle):.1f} deg, score {predictions[0]["score"]:.3f}'
            import textwrap
            lines=textwrap.wrap(caption,width=max(25,image.width//7))
            canvas=Image.new('RGB',(image.width,image.height+18*len(lines)+12),'white')
            canvas.paste(tile,(0,18*len(lines)+12))
            ImageDraw.Draw(canvas).multiline_text((6,5),'\n'.join(lines),fill=(20,30,45))
            canvas.save(self.output/f'{key}_{suffix}.png')
        self.cards.append(f'<article><p>{html.escape(row["text"])}</p><img src="{key}_prediction.png" alt="Predicted grasp rectangles"><img src="{key}_heatmap.png" alt="Quality heatmap and predicted grasps"><code>{html.escape(row["sample_id"])}</code></article>')

    def finish(self):
        (self.output/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SwapGrasp predictions</title><style>body{font:16px system-ui;background:#f4f6fa;color:#172136;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}article{background:white;padding:16px;overflow-wrap:anywhere}img{width:49%;vertical-align:top}</style><h1>SwapGrasp predictions</h1><p>Red: top-1. Amber: other candidates. Heatmap color uses a fixed score range [0,1].</p><main>'''+''.join(self.cards)+'</main></html>',encoding='utf-8')


def overlay(image, grasps, predictions=None, quality=None):
    image=image.convert('RGB').copy()
    if quality is not None:
        from matplotlib import colormaps
        # One fixed [0,1] scale for all panels; never per-image min-max.
        heat=(colormaps['magma'](np.clip(quality,0,1))[:,:,:3]*255).astype(np.uint8)
        image=Image.blend(image,Image.fromarray(heat).resize(image.size),.4)
    draw=ImageDraw.Draw(image)
    for g in grasps:
        pts=[tuple(p) for p in corners(g)]
        draw.line(pts+[pts[0]],fill=(55,220,100),width=2)
    for i,p in enumerate(predictions or []):
        pts=[tuple(p) for p in corners(p['grasp'])]
        draw.line(pts+[pts[0]],fill=(255,80,65) if i==0 else (255,195,60),width=3 if i==0 else 1)
    return image


def image_data(image):
    buf=io.BytesIO(); image.save(buf,format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()


def gallery(dataset, predictions, output, limit=100):
    pred={p['sample_id']:p for p in predictions}; cards=[]
    for row in dataset.rows[:limit]:
        ps=pred.get(row['sample_id'],{}).get('predictions',[])
        success=grasp_success(ps,row['grasps']) if predictions else None
        with Image.open(dataset.path(row['image'])) as im:
            image=overlay(im,row['grasps'],ps)
        if image.width>600:
            image.thumbnail((600,600))
        status='unscored' if success is None else ('success' if success else 'failure')
        cards.append(f'<article data-status="{status}"><img alt="GT and predicted grasp overlay" src="{image_data(image)}"><p>{html.escape(row["text"])}</p><code>{html.escape(row["sample_id"])}</code><p>{status}</p></article>')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SwapGrasp prediction gallery</title><style>body{font:16px system-ui;margin:24px;background:#f4f6fa;color:#172136}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}article{background:white;padding:12px;overflow-wrap:anywhere}img{width:100%}select{padding:8px;margin:16px 0}</style><h1>SwapGrasp</h1><p>Green: all GT. Red: top-1. Amber: other predictions. Synthetic samples are engineering checks only.</p><label>Filter <select id="filter"><option value="all">All</option><option>success</option><option>failure</option><option>unscored</option></select></label><main>'''+''.join(cards)+'''</main><script>document.getElementById('filter').addEventListener('change',e=>{document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.status!==e.target.value)})</script></html>''',encoding='utf-8')


def curves(log_file, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import json
    rows=[json.loads(s) for s in Path(log_file).read_text().splitlines() if s.strip()]
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    axs=axs.ravel()
    points=[(r['step'],r['loss']) for r in rows if 'loss' in r]
    if points:axs[0].plot(*zip(*points),label='total objective',color='#0072B2')
    axs[0].set(xlabel='Optimizer step',ylabel='Total training loss',title='Optimization')
    axs[0].set_yscale('log');axs[0].set_ylabel('Total loss (log scale)')
    for key,label,color in [('quality','grasp quality','#009E73'),('geometry','grasp geometry','#E69F00'),
                            ('swap','raw L-swap','#D55E00'),('weighted_swap','weighted L-swap','#CC79A7')]:
        points=[(r['step'],r[key]) for r in rows if key in r]
        if points:axs[1].plot(*zip(*points),label=label,color=color)
    axs[1].set(xlabel='Optimizer step',ylabel='Component loss',title='Loss components')
    axs[1].set_yscale('log');axs[1].set_ylabel('Component loss (log scale; zero omitted)')
    axs[1].legend()
    for key in ['gpu_peak_allocated_gb','gpu_peak_reserved_gb']:
        points=[(r['step'],r[key]) for r in rows if key in r]
        if points:axs[2].plot(*zip(*points),label=key.replace('gpu_peak_','peak '))
    axs[2].set(xlabel='Optimizer step',ylabel='Peak memory (GB)',title='GPU memory')
    axs[2].legend()
    for key in ['val/grasp_at1','val/grasp_at5','val/pair_both_grasp_at1']:
        points=[(r['step'],r[key]) for r in rows if r.get(key) is not None]
        if points:
            axs[3].plot(*zip(*points),label=key)
    axs[3].set(xlabel='Optimizer step',ylabel='Validation success',ylim=(0,1),title='Validation')
    if axs[3].lines:axs[3].legend()
    fig.tight_layout(); Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=160); fig.savefig(str(Path(output).with_suffix('.svg'))); plt.close(fig)

