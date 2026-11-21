import json
from pathlib import Path
import subprocess
import shutil
import sys

root=Path('data')
candidates=json.loads((root/'segments_candidates.json').read_text())['segments']
fixed=Path('protocol/review_decisions.json')
if not (root/'review_decisions.json').exists():shutil.copyfile(fixed,root/'review_decisions.json')
decisions=json.loads((root/'review_decisions.json').read_text())
expected=json.loads(fixed.read_text())
assert {k:v['status'] for k,v in decisions.items()}=={k:v['status'] for k,v in expected.items()}, 'Controlled comparison requires identical review decisions'
assert {s['segment_id'] for s in candidates}==set(decisions), 'Candidate IDs changed; review must be reconciled'
subprocess.run([sys.executable,'-m','memorymaze_lewm','review','--segments','data/segments_candidates.json',
    '--decisions','data/review_decisions.json','--out','data/segments_reviewed.json'],check=True)
subprocess.run([sys.executable,'scripts/segment_statistics.py'],check=True)
print('REVIEW_AND_DISTRIBUTION_CHECK_COMPLETE',flush=True)
