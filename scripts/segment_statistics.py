"""Describe reviewed conditions without reading model errors or selecting by performance."""
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memorymaze_lewm.data import load_episode

root=Path('data');records=json.loads((root/'segments_reviewed.json').read_text())['segments']
missing=json.loads((root/'segments_candidates.json').read_text())['rejected']
result={'visibility_policy':'red-pixel heuristic; future object masks are not ground truth',
        'selection_policy':'initial-frame review only; no model error used for inclusion','splits':{}}
for split in ['probe_train','probe_val','test']:
    candidates=[s for s in records if s['split']==split]
    approved=[s for s in candidates if s['review_status']=='approved']
    values=[];actions=np.zeros(6)
    for s in approved:
        d=load_episode(s['path']);t=s['origin'];end=t+50
        v=d['targets_vec'][t,s['object_id']];position=d['agent_pos'][t:end+1]
        direction=d['agent_dir'][t:end+1];yaw=np.unwrap(np.arctan2(direction[:,1],direction[:,0]))
        longest=run=0
        for visibility in s['visibility'][1:]:
            run=run+1 if visibility=='hidden' else 0;longest=max(longest,run)
        values.append([float(np.linalg.norm(v)),float(v[0]),float(v[1]),
            float(np.linalg.norm(np.diff(position,axis=0),axis=1).sum()),
            float(np.abs(np.diff(yaw)).sum()*180/np.pi),s['first_hidden_step'],longest])
        actions+=d['action'][t+1:end+1].sum(0)
    fields=['initial_distance','initial_lateral','initial_forward','travel_distance_50',
            'total_turn_degrees_50','first_hidden_step_heuristic','longest_hidden_run_heuristic']
    a=np.asarray(values)
    result['splits'][split]={'candidate_episodes':len(candidates),'no_candidate_episodes':sum(x['split']==split for x in missing),
        'approved':len(approved),'rejected':sum(s['review_status']=='rejected' for s in candidates),
        'pending':sum(s['review_status']=='pending' for s in candidates),
        'review_methods':dict(Counter(s.get('reviewer','inherited recorded review') for s in approved)),
        'condition_quantiles_min_q25_median_q75_max':{k:np.quantile(a[:,i],[0,.25,.5,.75,1]).tolist() for i,k in enumerate(fields)} if len(a) else {},
        'recorded_action_counts_in_order_0_to_5':actions.tolist()}
(root/'segment_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{f:v[f] for f in ['candidate_episodes','no_candidate_episodes','approved','rejected','pending']} for k,v in result['splits'].items()}),flush=True)
