"""Finish the paper and packages after the independent training/evaluation workflow."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parents[1];run=Path(sys.argv[1]).resolve();os.chdir(root)
state_path=run/'postprocess_state.json'
state_path.write_text(json.dumps({'status':'waiting_for_final_evaluation'}))
while True:
    path=run/'workflow_state.json'
    if path.exists():
        state=json.loads(path.read_text())
        if state['status'].startswith('complete'):break
    time.sleep(45)
commands=[('paper',[sys.executable,'scripts/update_paper_results.py','--run',str(run),'--paper',str(root/'paper')]),
          ('package',[sys.executable,'scripts/package_submission.py','--run',str(run)])]
errors={}
for name,command in commands:
    state_path.write_text(json.dumps({'status':name,'errors':errors}))
    result=subprocess.run(command,cwd=root,capture_output=True,text=True)
    (run/(name+'_finalize.log')).write_text(result.stdout+'\n'+result.stderr)
    if result.returncode:errors[name]=(result.stdout+'\n'+result.stderr)[-3000:]
state_path.write_text(json.dumps({'status':'complete' if not errors else 'complete_with_errors','errors':errors},indent=2))
print(state_path.read_text(),flush=True)
