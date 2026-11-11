"""Print a compact download/training snapshot without loading CUDA."""
import json
import os
from pathlib import Path
import sys
import time

root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from download_data import FILES

catalog_path=root/'data/downloads/catalog.json';archives=[]
if catalog_path.exists():
    catalog=json.loads(catalog_path.read_text())
    for repo,names in FILES.items():
        for name in names:
            path=root/'data/downloads'/repo.split('/')[-1]/name
            expected=catalog[repo]['files'][name]['size']
            actual=path if path.exists() else path.with_name(path.name+'.part')
            received=actual.stat().st_size if actual.exists() else 0
            archives.append({'name':name,'bytes':received,'total_bytes':expected,
                             'percent':round(100*received/expected,1),
                             'sha256_verified':path.with_name(path.name+'.verified.json').exists()})
result={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'archives':archives}
if archives:
    result['download_percent']=round(100*sum(r['bytes'] for r in archives)/sum(r['total_bytes'] for r in archives),1)
for name in ['download','real_pilot','real_overfit_1500']:
    pidfile=root/f'artifacts/{name}.pid'
    if pidfile.exists():
        pid=int(pidfile.read_text());alive=True
        try:os.kill(pid,0)
        except ProcessLookupError:alive=False
        result[name+'_process']={'pid':pid,'alive':alive}
for folder in ['real_overfit_1500','real_pilot_001']:
    path=root/'runs'/folder/'state.json'
    if path.exists():
        state=json.loads(path.read_text())
        result[folder]={k:state[k] for k in ['status','active','scope','training_metrics'] if k in state}
        for method in ['overfit','baseline','swap']:
            log=root/'runs'/folder/method/'metrics.jsonl'
            if log.exists():
                lines=log.read_text().splitlines()
                if lines:result[folder][method+'_latest']=json.loads(lines[-1])
print(json.dumps(result,indent=2,ensure_ascii=False))
(root/'artifacts').mkdir(exist_ok=True)
temp=root/'artifacts/progress.json.tmp';temp.write_text(json.dumps(result,indent=2))
temp.replace(root/'artifacts/progress.json')
