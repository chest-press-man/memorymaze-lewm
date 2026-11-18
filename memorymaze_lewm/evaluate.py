from pathlib import Path

import numpy as np

from .data import HORIZONS
from .io import read_json, write_json
from .probe import balanced_weights,fit_ridge,predict


def read_latents(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def episode_metrics(errors,episodes,horizons):
    result={}
    for h in HORIZONS:
        mask=horizons==h;means=[float(errors[mask&(episodes==e)].mean()) for e in np.unique(episodes[mask])]
        result[str(h)]={'mean_l2':float(np.mean(means)) if means else None,
                       'episode_std':float(np.std(means,ddof=1)) if len(means)>1 else None,
                       'episodes':len(means),'segments':int(mask.sum())}
    return result


def run_probe(latent_dir,out_dir):
    root=Path(latent_dir);out=Path(out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new probe directory')
    out.mkdir(parents=True,exist_ok=True)
    train=read_latents(root/'probe_train.npz');val=read_latents(root/'probe_val.npz');test=read_latents(root/'test.npz')
    ids=[set(d['episode']) for d in [train,val,test]]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('Probe/test episode leakage')
    w=balanced_weights(train['episode'],train['horizon']);scores=[];best=None;best_score=np.inf
    for lam in [0,1e-4,1e-3,1e-2,.1,1]:
        model=fit_ridge(train['latent'],train['target'],w,lam)
        errors=np.linalg.norm(predict(model,val['latent'])-val['target'],axis=-1)
        metrics=episode_metrics(errors,val['episode'],val['horizon'])
        score=np.mean([v['mean_l2'] for v in metrics.values()])
        scores.append({'lambda':lam,'validation_mean_l2':float(score)})
        if score<=best_score:best_score=score;best=model
    np.savez_compressed(out/'probe.npz',**best)
    # Reload readout before test; LeWM and readout both stay frozen.
    with np.load(out/'probe.npz',allow_pickle=False) as f:best={k:f[k] for k in f.files}
    predictions=predict(best,test['latent']);errors=np.linalg.norm(predictions-test['target'],axis=-1)
    constant=(w[:,None]*train['target']).sum(0)
    baseline=np.linalg.norm(constant-test['target'],axis=-1)
    increase=np.empty_like(errors)
    for seg in np.unique(test['segment']):
        mask=test['segment']==seg;one=errors[mask&(test['horizon']==1)]
        if len(one)!=1:raise ValueError('Missing/duplicate 1-step reference')
        increase[mask]=errors[mask]-one[0]
    main=episode_metrics(errors,test['episode'],test['horizon'])
    ref=episode_metrics(baseline,test['episode'],test['horizon'])
    growth=episode_metrics(increase,test['episode'],test['horizon'])
    result={'selected_lambda':float(best['lambda']),'validation_scores':scores,'model':main,'constant_reference':ref,
            'increase_from_step1':growth,'constant_coordinate':constant.tolist(),
            'hidden_heuristic':episode_metrics(errors[test['visibility']=='hidden'],test['episode'][test['visibility']=='hidden'],test['horizon'][test['visibility']=='hidden']),
            'provenance':read_json(root/'provenance.json'),'coordinate_units':'Memory Maze normalized grid-cell coordinates; two ego-relative components',
            'probe_train_episodes':len(ids[0]),'probe_val_episodes':len(ids[1]),'test_episodes':len(ids[2])}
    write_json(out/'metrics.json',result)
    np.savez_compressed(out/'predictions.npz',prediction=predictions,target=test['target'],error=errors,**{k:test[k] for k in ['episode','segment','horizon','source_step','object_id','visibility','review_status']})
    with (out/'metrics.csv').open('w') as f:
        f.write('horizon,model_mean_l2,constant_mean_l2,increase_from_step1,episode_std,episodes,segments\n')
        for h in HORIZONS:
            a=main[str(h)];f.write(f'{h},{a["mean_l2"]},{ref[str(h)]["mean_l2"]},{growth[str(h)]["mean_l2"]},{a["episode_std"]},{a["episodes"]},{a["segments"]}\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(HORIZONS,[main[str(h)]['mean_l2'] for h in HORIZONS],'o-',label='LeWM + linear probe')
    ax.plot(HORIZONS,[ref[str(h)]['mean_l2'] for h in HORIZONS],'s--',label='Train-mean coordinate')
    ax.set(xlabel='Future prediction step',ylabel='Mean position L2 error (grid cells)',title='Memory Maze: exploratory pipeline evaluation')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'position_error.png',dpi=160);plt.close(fig)
    return result
