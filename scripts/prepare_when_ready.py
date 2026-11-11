"""Background worker: prepare and audit the real pilot once labels are verified.

Does not start real model training. Inspection contact sheets are written first.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1];os.chdir(root)
required=['Grasp-Anything/scene_description.zip','Grasp-Anything-pp/grasp_instructions.zip',
          'Grasp-Anything-pp/grasp_label_positive.zip','Grasp-Anything-pp/part_mask.zip']
state_path=Path('artifacts/real_preparation_state.json')
deadline=time.monotonic()+12*3600
while time.monotonic()<deadline:
    missing=[p for p in required if not Path('data/downloads',p+'.verified.json').exists()]
    if not missing:break
    state_path.write_text(json.dumps({'status':'waiting_for_verified_annotations','missing':missing},indent=2))
    time.sleep(30)
else:
    state_path.write_text(json.dumps({'status':'timeout_waiting_for_annotations'}));raise SystemExit(1)
commands=[
 [sys.executable,'prepare_real.py','--train-images','100','--val-images','20','--test-images','40'],
 [sys.executable,'tools.py','audit','--manifest','data/processed/manifest.jsonl','--output','artifacts/real_data_audit.json'],
 [sys.executable,'visualize.py','--manifest','data/processed/manifest.jsonl','--split','train','--output','artifacts/real_train_contact_sheet.html'],
 [sys.executable,'visualize.py','--manifest','data/processed/manifest.jsonl','--split','val','--output','artifacts/real_val_contact_sheet.html']]
state_path.write_text(json.dumps({'status':'preparing_pilot_subset'}))
for command in commands:
    print('RUN',' '.join(command),flush=True);r=subprocess.run(command)
    if r.returncode:
        state_path.write_text(json.dumps({'status':'failed','command':command,'exit_code':r.returncode},indent=2));raise SystemExit(r.returncode)
state_path.write_text(json.dumps({'status':'pilot_prepared_audit_passed_requires_visual_inspection',
                                 'manifest':'data/processed/manifest.jsonl','training_started':False},indent=2))
