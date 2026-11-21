import time
from pathlib import Path

import numpy as np
import torch

from .data import HORIZONS, cached_episode
from .io import read_json, sha256, write_json
from .model import load_adapter


def extract(checkpoint,segments_path,out_dir,device='cpu',factory=None,allow_unreviewed=False,max_step=50,threads=3):
    torch.set_num_threads(threads)
    if max_step<5:raise ValueError('max_step must be at least 5')
    horizons=list(range(max_step+1))
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
            if t<2 or t+max_step>=len(d['image']):raise ValueError('Segment is too short for requested max_step')
            if len(seg['visibility'])<=max_step:raise ValueError('Regenerate segments for requested max_step')
            images=d['image'][t-2:t+1]
            initial=adapter.initial_latent(images)
            future=adapter.rollout(images,d['action'][t-1:t+1].astype(np.float32),d['action'][t+1:t+max_step+1].astype(np.float32))
            if future.ndim!=2 or future.shape[0]!=max_step or initial.shape!=future.shape[1:]:raise ValueError('Initial/future latent shape mismatch')
            z=np.concatenate([initial[None,:],future],axis=0)
            if device=='cuda':torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            if z.ndim!=2 or z.shape[0]!=max_step+1 or not np.isfinite(z).all():raise ValueError('Adapter must return finite aligned initial/future latents')
            for h in horizons:
                groups[seg['split']].append({'latent':z[h],'target':d['targets_vec'][t+h,oid],
                    'episode':seg['episode_id'],'segment':seg['segment_id'],'horizon':h,'source_step':t+h,
                    'object_id':oid,'latent_source':'observed' if h==0 else 'predicted','visibility':seg['visibility'][h],'review_status':seg['review_status']})
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
                'factory':factory or 'memorymaze_lewm.model:LeWMAdapter','device':device,'horizons':horizons,'max_step':max_step,'horizon_zero_kind':'observed initial latent; not a future prediction',
                'input_policy':'initial RGB 3 frames only; no future RGB/GT passed to adapter',
                'claim':'exploratory: contains unreviewed candidates' if any(x['review_status']!='approved' for x in records) else 'reviewed initial contexts; per-step visibility may remain heuristic',
                'attempted_segments':len(records),'successful_segments':len(timings),'failures':failures,'timings':timings}
    write_json(out/'provenance.json',provenance)
    return provenance
