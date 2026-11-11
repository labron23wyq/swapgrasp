"""Replace draft-only results with completed, traceable three-seed measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory


def pct(x):return f'{100*x:.1f}'


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--paper',required=True);a=p.parse_args()
    run=Path(a.run);paper=Path(a.paper);ev=json.loads((run/'evidence.json').read_text())
    summary=ev['summary'];protocol=ev['protocol']
    if len(ev['test_metrics'])!=6 or not all(ev['same_schedule'].values()):
        raise ValueError('The final paper requires all six evaluated runs and matching schedules')
    baseline=summary['baseline']['grasp_at1'];swap=summary['swap']['grasp_at1']
    delta=100*(swap['mean']-baseline['mean'])
    abstract=r'''When several objects are graspable, a detector must choose the one named by the instruction. SwapGrasp adds a symmetric ranking loss over grasp-centre supports for two targets in the same image. On an image-disjoint Grasp-Anything++ subset, three matched seeds yield mean Top-1 success of '''+f"{pct(baseline['mean'])}\\% for the baseline and {pct(swap['mean'])}\\% for SwapGrasp. "+r'''We report paired-instruction success and stricter geometric thresholds to distinguish target selection from single-query grasp accuracy.'''
    methods=['Baseline','SwapGrasp'];table=[]
    for method,label in zip(('baseline','swap'),methods):
        m=summary[method]
        values=[m[k] for k in ('grasp_at1','grasp_at5','strict_at1','pair_both_grasp_at1')]
        table.append(label+' & '+' & '.join(pct(v['mean'])+r'$\pm$'+f"{100*v['sample_std']:.1f}" for v in values)+r'\\')
    comparison=[]
    for seed,cmp in ev['paired_comparisons'].items():
        c=cmp['grasp_at1'];lo,hi=c['image_cluster_bootstrap_95ci']
        comparison.append(f"{seed}: ${100*c['difference_b_minus_a']:+.2f}\\;[{100*lo:+.2f}, {100*hi:+.2f}]$")
    allpositive=all(c['grasp_at1']['image_cluster_bootstrap_95ci'][0]>0 for c in ev['paired_comparisons'].values())
    interpretation=('All intervals support a gain on this subset.' if allpositive
                    else 'These intervals do not establish a consistent gain.')
    abstract+=(' The loss improves Top-1 consistently in this protocol.' if allpositive
               else ' The loss does not establish a consistent Top-1 gain.')
    conclusion=('Same-image ranking makes the intended target explicit in the objective and improves Top-1 across matched seeds on this subset.'
                if allpositive else 'Same-image ranking makes the intended target explicit in the objective, but this experiment does not establish a consistent test gain.')
    conclusion+=' Limited pair coverage and noisy target semantics motivate further study.'
    spec=ev['prompt_specificity']['summary']
    interpretation+=f" The supplementary target-specificity gap changes from {pct(spec['baseline']['gap_at1']['mean'])} to {pct(spec['swap']['gap_at1']['mean'])} percentage points."
    experiments=r'''\section{Experiments}
\paragraph{Data.} We filter the official object split for usable positive annotations, remove train/test image overlap, and reserve validation images from training. This leaves 7,477/704/605 images and 25,701/2,486/1,960 prompts in train/validation/test. We exclude 362 queries with ambiguous box units. The test set contains 189 pairs from 36 images. Results apply to this subset.

\paragraph{Training.} Each seed (42, 123, 3407) runs both models for 5,050 steps on A100 GPUs, with BF16 and AdamW ($3\times10^{-4}$ learning rate, $10^{-4}$ decay). Each batch has 256 queries, half from 64 pairs. Both models use the same validation-based learning-rate rule and select checkpoints by validation Top-1.

\paragraph{Metrics.} A success requires one target GT with rotated IoU $>0.25$ and angle error $<30^\circ$. Top-1 checks the first candidate and Top-5 checks any of the five. Strict@1 uses $(0.5,15^\circ)$; Pair@1 requires Top-1 success for both instructions. The specificity gap subtracts competing-target hits from own-target hits over both directions of each pair.

\begin{table}[t]
\centering\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\toprule
Method & Top-1 & Top-5 & Strict@1 & Pair@1\\
\midrule
'''+ '\n'.join(table)+r'''
\bottomrule
\end{tabular}
\caption{Test success (\%), mean $\pm$ sample standard deviation across three seeds.}
\label{tab:results}
\end{table}

\paragraph{Results.} Table~\ref{tab:results} shows a mean Top-1 difference of '''+f'${delta:+.2f}$'+r''' points. Seed-wise differences and 95\% intervals from 1,000 paired image bootstrap resamples are '''+'; '.join(comparison)+'. '+interpretation+r'''

\paragraph{Limitations.} Generated language labels can mismatch objects. Only 386 training images supply pairs. Continuous polygon IoU is not calibrated against the official rasterized evaluator, and rectangle overlap does not test physical execution.

\section{Conclusion}
'''+conclusion+r'''

\noindent\textbf{Code:} GitHub URL to be added before submission.
'''
    source=(paper/'paper.tex').read_text()
    source=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:r'\begin{abstract}'+'\n'+abstract+'\n'+r'\end{abstract}',source,flags=re.S)
    start=source.index(r'\section{Experiments}');end=source.index(r'{\small',start)
    source=source[:start]+experiments+'\n'+source[end:]
    with TemporaryDirectory(prefix='build_',dir=paper) as directory:
        staging=Path(directory).resolve()
        for path in paper.iterdir():
            if path.is_file() and path.suffix in ('.sty','.bib','.bst','.png') and not path.name.startswith('page-'):
                shutil.copy2(path,staging/path.name)
        (staging/'paper.tex').write_text(source)
        compile_tex=['pdflatex','-interaction=nonstopmode','-halt-on-error','paper.tex']
        for command in [compile_tex,['bibtex','paper'],compile_tex,compile_tex]:
            result=subprocess.run(command,cwd=staging,capture_output=True,text=True)
            if result.returncode:raise RuntimeError((result.stdout+'\n'+result.stderr)[-4000:])
        info=subprocess.run(['pdfinfo',str(staging/'paper.pdf')],capture_output=True,text=True,check=True).stdout
        pages=int(re.search(r'Pages:\s+(\d+)',info)[1])
        if pages>2:raise ValueError(f'Paper is {pages} pages; shorten before submission')
        subprocess.run(['pdftoppm','-scale-to','1500','-png',str(staging/'paper.pdf'),str(staging/'page')],check=True)
        for name in ('paper.tex','paper.pdf','paper.log',*(f'page-{i+1}.png' for i in range(pages))):
            (staging/name).replace(paper/name)
    (paper/'result_provenance.json').write_text(json.dumps({'source':str(run/'evidence.json'),'page_count':pages,
                                                           'evidence_sha256':hashlib.sha256((run/'evidence.json').read_bytes()).hexdigest(),
                                                           'manual_visual_review_required':True},indent=2))
    print('UPDATED PAPER',paper/'paper.pdf','pages',pages,flush=True)


if __name__=='__main__':main()
