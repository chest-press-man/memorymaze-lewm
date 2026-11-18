import time
from pathlib import Path

import numpy as np
import torch

from .data import HORIZONS, cached_episode
from .io import read_json, sha256, write_json
from .model import load_adapter


def extract(checkpoint,segments_path,out_dir,device='cpu',factory=None,allow_unreviewed=False):
    torch.set_num_threads(4)
    out=Path(out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new rollout directory')
    out.mkdir(parents=True,exist_ok=True)
    records=read_json(segments_path)['segments']
    approved=[x for x in records if x['review_status']=='approved']
    if not allow_unreviewed:
        records=approved
        if not records:raise ValueError('No reviewed segments. Review candidates first; unreviewed use must be explicit.')
    for split in ['probe_train','probe_val','test']:
        if not any(x['split']==split for x in records):raise ValueError(f'No usable {split} segments')
    adapter=load_adapter(checkpoint,device,factory)
    groups={s:[] for s in ['probe_train','probe_val','test']};failures=[];timings=[]
    for seg in records:
        d=cached_episode(seg['path']);t=seg['origin'];oid=seg['object_id']
        # Deliberately pass no future pixels or GT to any model adapter.
        started=time.perf_counter()
        try:
            z=adapter.rollout(d['image'][t-2:t+1],d['action'][t-1:t+1].astype(np.float32),d['action'][t+1:t+51].astype(np.float32))
            if device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            if z.ndim!=2 or z.shape[0]!=50 or not np.isfinite(z).all():raise ValueError('Adapter must return finite (50,D) future latents')
            for h in HORIZONS:
                groups[seg['split']].append({'latent':z[h-1],'target':d['targets_vec'][t+h,oid],
                    'episode':seg['episode_id'],'segment':seg['segment_id'],'horizon':h,'source_step':t+h,
                    'object_id':oid,'visibility':seg['visibility'][h],'review_status':seg['review_status']})
            timings.append({'segment_id':seg['segment_id'],'seconds':elapsed})
            print(f'rollout {seg["split"]}: {seg["segment_id"]} {elapsed:.2f}s',flush=True)
        except Exception as exc:
            failures.append({'segment_id':seg['segment_id'],'split':seg['split'],'error':repr(exc)})
            write_json(out/'failures.json',failures)
    for split,rows in groups.items():
        if not rows:raise RuntimeError(f'All {split} rollouts failed; see failures.json')
        np.savez_compressed(out/f'{split}.npz',**{key:np.stack([r[key] for r in rows]) for key in rows[0]})
    provenance={'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_hash':sha256(checkpoint),
                'segments_hash':sha256(segments_path),'segments_path':str(Path(segments_path).resolve()),
                'factory':factory or 'memorymaze_lewm.model:LeWMAdapter','device':device,'horizons':HORIZONS,
                'input_policy':'initial RGB 3 frames only; no future RGB/GT passed to adapter',
                'claim':'exploratory: contains unreviewed candidates' if any(x['review_status']!='approved' for x in records) else 'reviewed initial contexts; per-step visibility may remain heuristic',
                'attempted_segments':len(records),'successful_segments':len(timings),'failures':failures,'timings':timings}
    write_json(out/'provenance.json',provenance)
    return provenance
