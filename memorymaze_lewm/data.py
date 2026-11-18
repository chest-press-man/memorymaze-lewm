from functools import lru_cache
from pathlib import Path

import numpy as np

from .io import read_json, write_json

HORIZONS=list(range(51))


def load_episode(path):
    with np.load(path,allow_pickle=False) as f:
        d={k:f[k] for k in f.files}
    n=len(d['image'])
    if d['image'].shape!=(n,64,64,3) or d['image'].dtype!=np.uint8:raise ValueError('Expected 64x64 RGB uint8')
    if d['action'].shape!=(n,6) or not np.all(d['action'][0]==0):raise ValueError('Invalid initial action')
    a=d['action'][1:]
    if not np.all((a==0)|(a==1)) or not np.all(a.sum(-1)==1):raise ValueError('Invalid one-hot actions')
    if d['targets_vec'].shape!=(n,3,2) or not np.isfinite(d['targets_vec']).all():raise ValueError('Invalid object coordinates')
    if not np.allclose(d['targets_pos'],d['targets_pos'][0]):raise ValueError('Objects moved')
    return d


@lru_cache(maxsize=6)
def cached_episode(path):
    return load_episode(path)


def make_window(d,start,history=3):
    stop=start+history+1
    if start<0 or stop>len(d['image']):raise ValueError('Window outside episode')
    if d['reset'][start+1:stop].any() or d['terminal'][start:stop-1].any():raise ValueError('Window crosses boundary')
    return d['image'][start:stop],d['action'][start+1:stop].astype(np.float32)


def preprocess(images,size=224):
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as F
    x=torch.from_numpy(np.array(images,copy=True)).movedim(-1,-3).float()/255
    x[...,:2,:]=0;x[...,-2:,:]=0;x[...,:,:2]=0;x[...,:,-2:]=0
    mean=x.new_tensor([.485,.456,.406]).view(3,1,1)
    std=x.new_tensor([.229,.224,.225]).view(3,1,1)
    # Match upstream ToImage(ImageNet) then Resize; never resize twice.
    return F.resize((x-mean)/std,[size,size],interpolation=InterpolationMode.BILINEAR,antialias=True)


def validate_splits(splits):
    seen=set()
    for name,paths in splits.items():
        ids=[Path(p).stem for p in paths]
        if len(set(ids))!=len(ids) or seen.intersection(ids):raise ValueError(f'Episode overlap in {name}')
        seen.update(ids)


def make_splits(data_root,counts=(100,20,30,10,30),protocol=False,expected_episode_ids=None):
    records=read_json(Path(data_root)/'acquisition.json')['episodes']
    train=sorted([str(Path(data_root).resolve()/'train'/Path(r['path']).name) for r in records if r['source']=='train'])
    test=sorted([str(Path(data_root).resolve()/'eval'/Path(r['path']).name) for r in records if r['source']=='eval'])
    if len(counts)!=5 or any(c<1 for c in counts):raise ValueError('Five positive split counts required')
    if len(train)<sum(counts[:4]) or len(test)<counts[4]:raise ValueError('Acquire enough episodes first')
    names=['model_train','model_val','probe_train','probe_val']
    if protocol:
        if counts[0]<500 or tuple(counts[1:])!=(100,300,100,200):raise ValueError('Scaled protocol fixes val/Probe/test at 100/300/100/200')
        # Preserve the original first 160 roles and the already-planned stage-1 membership.
        splits={};offset=0
        for name,count in zip(names,[100,20,30,10]):
            splits[name]=train[offset:offset+count];offset+=count
        for name,count in zip(names,[400,80,270,90]):
            splits[name]+=train[offset:offset+count];offset+=count
        splits['model_train']+=train[1000:1000+counts[0]-500]
    else:
        splits={};offset=0
        for name,count in zip(names,counts[:4]):
            splits[name]=train[offset:offset+count];offset+=count
    splits['test']=test[:counts[4]];validate_splits(splits)
    if expected_episode_ids is not None:
        actual={k:[Path(p).name for p in v] for k,v in splits.items()}
        if actual!=expected_episode_ids:raise ValueError('Episode membership differs from the previously reserved protocol; do not silently reassign')
    layouts={}
    for name,paths in splits.items():
        layouts[name]={cached_episode(p)['maze_layout'][0].tobytes().hex() for p in paths}
    overlaps={f'{a}/{b}':len(layouts[a]&layouts[b]) for a in layouts for b in layouts if a<b}
    result={'splits':splits,'counts':{k:len(v) for k,v in splits.items()},'layout_overlaps':overlaps,
            'layout_policy':'episode-disjoint; layout overlap measured, not silently assumed absent'}
    if protocol:
        result.update(protocol='fixed-holdouts-v2',normalizer_train=splits['model_train'][:500],normalizer_policy='fixed common 500 training episodes')
    write_json(Path(data_root)/'splits.json',result)
    return result


def action_stats(paths):
    total=np.zeros(6);squares=np.zeros(6);count=0
    for p in paths:
        a=cached_episode(p)['action'][1:];total+=a.sum(0);squares+=(a*a).sum(0);count+=len(a)
    mean=total/count;std=np.sqrt(np.maximum(squares/count-mean**2,0))
    return {'mean':mean.tolist(),'std':np.maximum(std,1e-6).tolist(),'count':count,'ddof':0}


def normalize_action(a,stats):
    return ((a-np.array(stats['mean']))/np.array(stats['std'])).astype(np.float32)
