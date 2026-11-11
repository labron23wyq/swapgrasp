import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import pytest


def test_result_updater_builds_two_page_pdf(tmp_path):
    root=Path(__file__).resolve().parents[1]
    if not (root/'paper/paper.tex').exists() or any(not shutil.which(x) for x in ('pdflatex','bibtex','pdfinfo','pdftoppm')):
        pytest.skip('Optional LaTeX/Poppler and paper sources required')
    paper=tmp_path/'paper';paper.mkdir()
    for name in ('paper.tex','references.bib','cvpr.sty','ieeenat_fullname.bst'):
        shutil.copy2(root/'paper'/name,paper/name)
    run=tmp_path/'run';run.mkdir()
    seeds=[42,123,3407]
    values=lambda mean:{'mean':mean,'sample_std':0.02,'n':3}
    summary={method:{key:values(mean) for key,mean in {
        'grasp_at1':0.4 if method=='baseline' else 0.45,
        'grasp_at5':0.7 if method=='baseline' else 0.75,
        'strict_at1':0.2 if method=='baseline' else 0.23,
        'pair_both_grasp_at1':0.1 if method=='baseline' else 0.12}.items()}
        for method in ('baseline','swap')}
    specificity={'summary':{method:{'gap_at1':values(0.3 if method=='baseline' else 0.4)}
                            for method in ('baseline','swap')}}
    evidence={'summary':summary,'protocol':{'images':8,'queries':24,
        'images_by_split':{'train':4,'val':2,'test':2},
        'queries_by_split':{'train':12,'val':6,'test':6},
        'pairs_by_split':{'train':4,'val':2,'test':2},'batch_size_queries':8,
        'pairs_per_step':2,'epochs_equivalent':50,'steps_per_epoch':2,
        'max_steps':100,'seeds':seeds},
        'test_metrics':[{} for _ in range(6)],'same_schedule':{str(seed):True for seed in seeds},
        'prompt_specificity':specificity,
        'paired_comparisons':{str(seed):{'grasp_at1':{'difference_b_minus_a':.05,
            'image_cluster_bootstrap_95ci':[.01,.09]}} for seed in seeds}}
    (run/'evidence.json').write_text(json.dumps(evidence))
    result=subprocess.run([sys.executable,'scripts/update_paper_results.py','--run',str(run),
                           '--paper',str(paper)],cwd=root,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    assert re.search(r'Pages:\s+2\b',subprocess.run(['pdfinfo',str(paper/'paper.pdf')],
                   check=True,text=True,capture_output=True).stdout)
    assert len(list(paper.glob('page-*.png')))==2
    assert 'result_provenance.json' in {p.name for p in paper.iterdir()}
