from pathlib import Path

import numpy as np

from .data import HORIZONS
from .io import read_json, write_json
from .probe import balanced_weights,fit_ridge,predict


def read_latents(path):
    with np.load(path,allow_pickle=False) as f:return {k:f[k] for k in f.files}


def episode_metrics(errors,episodes,horizons,steps=None):
    result={}
    for h in HORIZONS if steps is None else steps:
        mask=horizons==h;means=[float(errors[mask&(episodes==e)].mean()) for e in np.unique(episodes[mask])]
        result[str(h)]={'mean_l2':float(np.mean(means)) if means else None,
                       'episode_std':float(np.std(means,ddof=1)) if len(means)>1 else None,
                       'episodes':len(means),'segments':int(mask.sum())}
    return result


def decline_summary(errors,episodes,horizons):
    first=int(horizons.min());last=int(horizons.max());differences=[]
    for e in np.unique(episodes):
        a=errors[(episodes==e)&(horizons==first)]
        b=errors[(episodes==e)&(horizons==last)]
        if len(a) and len(b):differences.append(float(b.mean()-a.mean()))
    if not differences:raise ValueError('No paired validation episodes')
    d=np.asarray(differences);rng=np.random.default_rng(42)
    bootstrap=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)
    lo,hi=np.quantile(bootstrap,[.025,.975])
    return {'reference_step':first,'last_step':last,'paired_episodes':len(d),
            'mean_l2_increase':float(d.mean()),'bootstrap_95_percentile_ci':[float(lo),float(hi)],
            'decline_detected':bool(lo>0),'selection_data':'probe validation only',
            'interpretation':'No detected decline is not evidence of equivalence; longer rollout is a separate future protocol.'}


def run_probe(latent_dir,out_dir):
    root=Path(latent_dir);out=Path(out_dir)
    if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new probe directory')
    out.mkdir(parents=True,exist_ok=True)
    train=read_latents(root/'probe_train.npz');val=read_latents(root/'probe_val.npz');test=read_latents(root/'test.npz')
    ids=[set(d['episode']) for d in [train,val,test]]
    if any(ids[i]&ids[j] for i in range(3) for j in range(i)):raise ValueError('Probe/test episode leakage')
    steps=sorted(int(h) for h in np.unique(train['horizon']))
    if any(sorted(np.unique(d['horizon']).tolist())!=steps for d in [val,test]):raise ValueError('Split step grids differ')
    for d in [train,val,test]:
        for segment in np.unique(d['segment']):
            if sorted(d['horizon'][d['segment']==segment].tolist())!=steps:raise ValueError('Missing/duplicate step within segment')
        if not np.isfinite(d['latent']).all() or not np.isfinite(d['target']).all():raise ValueError('Nonfinite Probe data')
    w=balanced_weights(train['episode'],train['horizon']);scores=[];best=None;best_score=np.inf
    for lam in [0,1e-4,1e-3,1e-2,.1,1]:
        model=fit_ridge(train['latent'],train['target'],w,lam)
        errors=np.linalg.norm(predict(model,val['latent'])-val['target'],axis=-1)
        metrics=episode_metrics(errors,val['episode'],val['horizon'],steps)
        score=np.mean([v['mean_l2'] for v in metrics.values()])
        scores.append({'lambda':lam,'validation_mean_l2':float(score)})
        if score<=best_score:best_score=score;best=model
    np.savez_compressed(out/'probe.npz',**best)
    with np.load(out/'probe.npz',allow_pickle=False) as f:best={k:f[k] for k in f.files}
    validation_errors=np.linalg.norm(predict(best,val['latent'])-val['target'],axis=-1)
    predictions=predict(best,test['latent']);errors=np.linalg.norm(predictions-test['target'],axis=-1)
    constant=(w[:,None]*train['target']).sum(0)
    baseline=np.linalg.norm(constant-test['target'],axis=-1)
    step_baseline=np.empty_like(errors)
    for h in steps:
        mask=train['horizon']==h;hw=w[mask]/w[mask].sum()
        mean=(hw[:,None]*train['target'][mask]).sum(0)
        mask_test=test['horizon']==h
        step_baseline[mask_test]=np.linalg.norm(mean-test['target'][mask_test],axis=-1)
    increase=np.empty_like(errors);first=steps[0]
    for seg in np.unique(test['segment']):
        mask=test['segment']==seg;reference=errors[mask&(test['horizon']==first)]
        if len(reference)!=1:raise ValueError('Missing/duplicate reference step')
        increase[mask]=errors[mask]-reference[0]
    main=episode_metrics(errors,test['episode'],test['horizon'],steps)
    ref=episode_metrics(baseline,test['episode'],test['horizon'],steps)
    growth=episode_metrics(increase,test['episode'],test['horizon'],steps)
    result={'selected_lambda':float(best['lambda']),'validation_scores':scores,'model':main,'constant_reference':ref,
            f'increase_from_step{first}':growth,'constant_coordinate':constant.tolist(),
            'step_mean_reference':episode_metrics(step_baseline,test['episode'],test['horizon'],steps),
            'hidden_heuristic':episode_metrics(errors[test['visibility']=='hidden'],test['episode'][test['visibility']=='hidden'],test['horizon'][test['visibility']=='hidden'],steps),
            'provenance':read_json(root/'provenance.json'),'coordinate_units':'Memory Maze grid-cell coordinates; two ego-relative components',
            'horizons':steps,'horizon_zero_kind':'observed initial latent; not a future prediction' if first==0 else 'not present',
            'validation_decline':decline_summary(validation_errors,val['episode'],val['horizon']),
            'probe_train_episodes':len(ids[0]),'probe_val_episodes':len(ids[1]),'test_episodes':len(ids[2])}
    write_json(out/'metrics.json',result)
    keys=['episode','segment','horizon','source_step','object_id','visibility','review_status']
    if 'latent_source' in test:keys.append('latent_source')
    np.savez_compressed(out/'predictions.npz',prediction=predictions,target=test['target'],error=errors,**{k:test[k] for k in keys})
    with (out/'metrics.csv').open('w') as f:
        f.write(f'horizon,model_mean_l2,constant_mean_l2,increase_from_step{first},episode_std,episodes,segments\n')
        for h in steps:
            a=main[str(h)];f.write(f'{h},{a["mean_l2"]},{ref[str(h)]["mean_l2"]},{growth[str(h)]["mean_l2"]},{a["episode_std"]},{a["episodes"]},{a["segments"]}\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5))
    ax.plot(steps,[main[str(h)]['mean_l2'] for h in steps],label='LeWM + linear probe')
    ax.plot(steps,[ref[str(h)]['mean_l2'] for h in steps],'--',label='Train-mean coordinate')
    ax.plot(steps,[result['step_mean_reference'][str(h)]['mean_l2'] for h in steps],':',label='Step-specific train mean')
    if first==0:ax.axvline(.5,color='grey',alpha=.5)
    ax.set(xlabel='Step (0 = observed initial state)',ylabel='Mean position L2 error (grid cells)',title='Memory Maze: fixed-model dense readout')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'position_error.png',dpi=160);plt.close(fig)
    return result
