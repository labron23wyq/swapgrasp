import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def test_package_requires_complete_runs_and_excludes_weights(tmp_path):
    source_root=Path(__file__).resolve().parents[1]
    root=tmp_path/'project'
    for folder in ('scripts','swapgrasp','tests','configs','paper','artifacts'):
        (root/folder).mkdir(parents=True,exist_ok=True)
        (root/folder/'placeholder.py').write_text('pass\n')
    shutil.copy2(source_root/'scripts/package_submission.py',root/'scripts/package_submission.py')
    (root/'README.md').write_text('Source package fixture\n')
    (root/'artifacts/requirements-resolved.txt').write_text('test-only\n')
    run=root/'runs/submission_001';run.mkdir(parents=True)
    names=[];directories={}
    for method in ('baseline','swap'):
        for seed in (42,123,3407):
            name=f'{method}_s{seed}';names.append(name);folder=run/name;folder.mkdir()
            (folder/'config.yaml').write_text('seed: '+str(seed)+'\n')
            directories[name]={'directory':str(folder)}
    manifest=root/'data/manifest.jsonl';manifest.parent.mkdir();manifest.write_text('{}\n')
    pairs=root/'data/train_pairs.jsonl';pairs.write_text('{}\n')
    for filename in ('manifest.jsonl','train_pairs.jsonl','protocol.json','adapter_summary.json','quarantine.jsonl'):
        (manifest.parent/filename).write_text('{}\n')
    protocol={'runs':directories,'seeds':[42,123,3407],'manifest':str(manifest),'pairs':str(pairs)}
    (run/'protocol.json').write_text(json.dumps(protocol))
    evidence={'test_metrics':{name:{} for name in names},'same_schedule':{'42':True,'123':True,'3407':True}}
    evidence_path=run/'evidence.json';evidence_path.write_text(json.dumps(evidence))
    (run/'workflow_state.json').write_text(json.dumps({'status':'complete','evaluated':names}))
    paper=root/'paper'
    (paper/'paper.pdf').write_bytes(b'%PDF mock')
    (paper/'paper.tex').write_text('\\end{document}\n')
    (paper/'result_provenance.json').write_text(json.dumps({'page_count':2,
        'evidence_sha256':hashlib.sha256(evidence_path.read_bytes()).hexdigest()}))
    first=Path(directories[names[0]]['directory'])
    (first/'best.pt').write_bytes(b'model weights')
    (first/'train.log').write_text('training log\n')
    (first/'test_predictions.jsonl').write_text('{}\n')
    (first/'test').mkdir();(first/'test/metrics.json').write_text('{}\n')
    (first/'test_visuals').mkdir();(first/'test_visuals/index.html').write_text('ok')
    (first/'test_visuals/image.jpg').write_bytes(b'dataset image')
    (run/'REPORT.zh-CN.md').write_text('# report\n')
    (run/'unexpected.bin').write_bytes(b'not distributable')

    result=subprocess.run([sys.executable,'scripts/package_submission.py','--run',str(run)],
                          cwd=root,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    output=root/'artifacts/submission_001'
    for name in ('source.zip','results.zip'):
        with zipfile.ZipFile(output/name) as archive:
            assert archive.testzip() is None
            members=set(archive.namelist())
            assert not any(member.endswith(('.pt','.jpg','.bin')) for member in members)
    with zipfile.ZipFile(output/'results.zip') as archive:
        members=set(archive.namelist())
        assert 'baseline_s42/test_predictions.jsonl' in members
        assert 'baseline_s42/test_visuals/index.html' in members
        assert 'paper/paper.pdf' in members
        assert 'unexpected.bin' not in members
    with zipfile.ZipFile(output/'source.zip') as archive:
        assert 'scripts/package_submission.py' in archive.namelist()
        assert 'configs/completed_swap_s3407.yaml' in archive.namelist()
    assert {path.name for path in output.iterdir()}=={'source.zip','source_sha256.json','results.zip','sha256.json'}
