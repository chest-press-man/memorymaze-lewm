import numpy as np


def balanced_weights(episodes,horizons):
    episodes=np.asarray(episodes);horizons=np.asarray(horizons);w=np.zeros(len(episodes))
    ids=np.unique(episodes)
    for e in ids:
        mask=episodes==e;hs=np.unique(horizons[mask])
        for h in hs:
            ix=mask&(horizons==h);w[ix]=1/len(ids)/len(hs)/ix.sum()
    return w


def fit_ridge(x,y,w,lam):
    x=np.asarray(x,np.float64);y=np.asarray(y,np.float64);w=np.asarray(w,np.float64);w=w/w.sum()
    mean=(w[:,None]*x).sum(0);std=np.sqrt((w[:,None]*(x-mean)**2).sum(0));std=np.maximum(std,1e-8)
    q=(x-mean)/std;ym=(w[:,None]*y).sum(0)
    a=q*np.sqrt(w[:,None]);b=(y-ym)*np.sqrt(w[:,None])
    if lam==0:coef=np.linalg.lstsq(a,b,rcond=1e-10)[0]
    else:coef=np.linalg.solve(a.T@a+lam*np.eye(a.shape[1]),a.T@b)
    return {'mean':mean,'std':std,'coef':coef,'bias':ym,'lambda':float(lam)}


def predict(model,x):
    return (np.asarray(x)-model['mean'])/model['std']@model['coef']+model['bias']
