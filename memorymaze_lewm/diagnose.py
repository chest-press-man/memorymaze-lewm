"""Separate observed-latent position diagnostic; never mixed with future rollout."""
from pathlib import Path

import numpy as np
import torch

from .data import cached_episode,preprocess
from .io import read_json,write_json
from .model import load_adapter
from .probe import fit_ridge,predict


def diagnose(checkpoint,segments_path,out_dir,device='cpu'):
    out=Path(out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new diagnostic directory')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3)
    adapter=load_adapter(checkpoint,device);groups={s:[] for s in ['probe_train','probe_val','test']}
    with torch.inference_mode():
        for seg in read_json(segments_path)['segments']:
            if seg['review_status']!='approved':continue
            d=cached_episode(seg['path']);t=seg['origin']
            adapter.reset_episode()
            pixels=preprocess(d['image'][t-2:t+1]).unsqueeze(0).to(device)
            z=adapter.model.encode({'pixels':pixels})['emb'][0,-1].cpu().numpy()
            groups[seg['split']].append((z,d['targets_vec'][t,seg['object_id']]))
    arrays={k:(np.stack([a[0] for a in v]),np.stack([a[1] for a in v])) for k,v in groups.items()}
    x,y=arrays['probe_train'];vx,vy=arrays['probe_val'];tx,ty=arrays['test']
    # Current candidate/review manifest uses one segment per episode.
    if any(len({s['episode_id'] for s in read_json(segments_path)['segments'] if s['split']==k and s['review_status']=='approved'})!=len(groups[k]) for k in groups):
        raise ValueError('Diagnostic currently requires one segment per episode')
    w=np.ones(len(x))/len(x);best=None;score=np.inf;selection=[]
    for lam in [0,1e-4,1e-3,1e-2,.1,1]:
        m=fit_ridge(x,y,w,lam);value=float(np.linalg.norm(predict(m,vx)-vy,axis=1).mean())
        selection.append({'lambda':lam,'validation_l2':value})
        if value<=score:best=m;score=value
    pred=predict(best,tx);constant=y.mean(0)
    result={'type':'actual observed latent diagnostic; not future prediction performance',
            'selected_lambda':best['lambda'],'validation_selection':selection,
            'test_mean_l2':float(np.linalg.norm(pred-ty,axis=1).mean()),
            'constant_mean_l2':float(np.linalg.norm(constant-ty,axis=1).mean()),
            'episode_counts':{k:len(v) for k,v in groups.items()}}
    np.savez_compressed(out/'diagnostic_probe.npz',**best)
    np.savez_compressed(out/'predictions.npz',prediction=pred,target=ty)
    write_json(out/'metrics.json',result)
    return result
