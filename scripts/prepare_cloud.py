"""Run data verification and fixed-protocol preparation inside a CPU Job."""
import subprocess
import sys

count=int(sys.argv[1]) if len(sys.argv)>1 else 500
if count not in [500,2000]:raise ValueError('Expected 500 or 2000 model-train episodes')
for args in [
    ['validate','--root','data','--reference','protocol/data_reference.json'],
    ['prepare','--root','data','--scaled-protocol','--model-train',str(count)],
    ['segments','--splits','data/splits.json','--max-step','50','--out','data/segments_candidates.json'],
]:
    subprocess.run([sys.executable,'-m','memorymaze_lewm']+args,check=True)
print('CLOUD_DATA_PREPARATION_COMPLETE',flush=True)
