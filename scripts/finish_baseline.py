"""Complete readout/evaluation/export for the selected, frozen baseline checkpoint."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memorymaze_lewm.data import load_episode
from memorymaze_lewm.io import read_json,sha256,write_json
from memorymaze_lewm.model import load_adapter
from memorymaze_lewm.probe import predict

run=Path(sys.argv[1]) if len(sys.argv)>1 else Path('runs/baseline500_001');best=run/'checkpoints/best.pt'
for args in [
 ['rollout','--checkpoint',str(best),'--segments','data/segments_reviewed.json','--device','cuda','--threads','3','--max-step','50','--out',str(run/'latents')],
 ['probe','--latents',str(run/'latents'),'--out',str(run/'probe')],
 ['diagnose','--checkpoint',str(best),'--segments','data/segments_reviewed.json','--device','cuda','--out',str(run/'observed_diagnostic')],
 ['export','--checkpoint',str(best),'--out',str(run/'inference.pt')],
]:subprocess.run([sys.executable,'-m','memorymaze_lewm']+args,check=True)
# Public inference artifacts retain provenance without private server paths.
checkpoint=torch.load(run/'inference.pt',map_location='cpu',weights_only=False)
checkpoint['initialization']={'source':'https://huggingface.co/quentinll/lewm-cube/tree/b0747c5002e86d2ce8f3cd8178004b97524c587d',
 'original_weights_sha256':'2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89',
 'original_action_dim':25,'reinitialized':['action_encoder'],'seed':42}
torch.save(checkpoint,run/'inference.pt')
path=read_json('data/splits.json')['splits']['model_val'][0];d=load_episode(path);t=10
args=(d['image'][t-2:t+1],d['action'][t-1:t+1].astype(np.float32),d['action'][t+1:t+51].astype(np.float32))
a=load_adapter(best,'cuda');b=load_adapter(run/'inference.pt','cuda')
delta=max(float(np.max(np.abs(a.initial_latent(args[0])-b.initial_latent(args[0])))),float(np.max(np.abs(a.rollout(*args)-b.rollout(*args)))))
assert delta<1e-6,delta
with np.load(run/'probe/probe.npz',allow_pickle=False) as f:probe={k:f[k] for k in f.files}
with np.load(run/'latents/test.npz',allow_pickle=False) as f:z=f['latent']
with np.load(run/'probe/predictions.npz',allow_pickle=False) as f:expected=f['prediction']
probe_delta=float(np.max(np.abs(predict(probe,z)-expected)))
assert probe_delta<1e-8,probe_delta
report={'selected_update':checkpoint['update'],'export_max_abs_difference_0_to_50':delta,
 'probe_reload_max_abs_difference':probe_delta,'best_sha256':sha256(best),
 'inference_sha256':sha256(run/'inference.pt'),'probe_sha256':sha256(run/'probe/probe.npz')}
write_json(run/'verification.json',report)
print('FINAL_STAGE_COMPLETED',json.dumps(report),flush=True)
