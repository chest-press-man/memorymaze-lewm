import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .data import action_stats, cached_episode, make_window, normalize_action, preprocess, validate_splits
from .io import read_json, sha256, write_json
from .model import create_model, initialize
from .vendor.sigreg import SIGReg


def make_batch(paths,rng,batch_size,stats,size):
    pixels=[];actions=[]
    for _ in range(batch_size):
        p=paths[int(rng.integers(len(paths)))];d=cached_episode(p)
        for attempt in range(50):
            start=int(rng.integers(len(d['image'])-3))
            try:images,acts=make_window(d,start);break
            except ValueError:
                if attempt==49:raise
        pixels.append(preprocess(images,size));actions.append(normalize_action(acts,stats))
    return torch.stack(pixels),torch.from_numpy(np.stack(actions))


def loss_values(model,pixels,actions,regularizer,weight):
    # encode accepts independent action length; no artificial padding is needed.
    reset=getattr(model,'reset_episode',None)
    if reset is not None:reset()
    out=model.encode({'pixels':pixels,'action':actions})
    pred=model.predict(out['emb'][:,:3],out['act_emb'])
    if pred.shape!=out['emb'][:,1:].shape:raise ValueError('Shifted target shape mismatch')
    mse=(pred-out['emb'][:,1:]).square().mean()
    reg=regularizer(out['emb'].transpose(0,1).float())
    return mse+weight*reg,mse,reg,out['emb'].detach().std(dim=(0,1)).mean()


def save_checkpoint(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def train(config_path,splits_path,run_dir,pretrained_dir=None,resume=False,model_factory=None,stop_after=None,init_checkpoint=None):
    if init_checkpoint and (resume or pretrained_dir):raise ValueError('init_checkpoint cannot be combined with pretrained or resume')
    cfg=read_json(config_path);splits=read_json(splits_path)['splits'];validate_splits(splits)
    if cfg['history']!=3 or cfg['frameskip']!=1 or cfg['batch_size']<2:raise ValueError('Expected history3/frameskip1/batch>=2')
    device=cfg['device']
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; choose CPU pilot explicitly')
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['seed'])
    run=Path(run_dir)
    if run.exists() and any(run.iterdir()) and not resume:raise FileExistsError('New run needs an empty output directory; use --resume to continue')
    run.mkdir(parents=True,exist_ok=True);(run/'checkpoints').mkdir(exist_ok=True)
    signature={'split_hash':sha256(splits_path),'config':cfg}
    if model_factory:signature['model_factory']=model_factory
    rng=np.random.default_rng(cfg['seed']);step=0;best=math.inf
    last=run/'checkpoints/last.pt'
    if resume:
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        if checkpoint['signature']!=signature:raise ValueError('Resume config/split mismatch')
        model_cfg=checkpoint['model_config'];model=create_model(model_cfg);model.load_state_dict(checkpoint['model'])
        stats=checkpoint['action_stats'];provenance=checkpoint['initialization'];step=checkpoint['update'];best=checkpoint['best_validation_mse']
    elif init_checkpoint:
        source=torch.load(init_checkpoint,map_location='cpu',weights_only=False)
        model_cfg=dict(source['model_config'])
        if model_factory:model_cfg['_model_factory']=model_factory
        model=create_model(model_cfg)
        if model_factory:
            current=model.state_dict()
            shared={k:v for k,v in source['model'].items() if k in current and current[k].shape==v.shape}
            if not shared:raise ValueError('No compatible warm-start parameters')
            result=model.load_state_dict(shared,strict=False)
            transfer={'new_keys':list(result.missing_keys),'transferred_keys':list(shared),
                      'unused_source_keys':[k for k in source['model'] if k not in shared]}
        else:model.load_state_dict(source['model'],strict=True);transfer={'reinitialized':[]}
        provenance={'source_checkpoint':str(Path(init_checkpoint).resolve()),'sha256':sha256(init_checkpoint),**transfer}
        stats=action_stats(splits['model_train'])
    else:model,model_cfg,provenance=initialize(pretrained_dir,model_factory);stats=action_stats(splits['model_train'])
    if resume and step>=cfg['updates']:
        print('Configured training already complete; existing artifacts preserved.',flush=True)
        return run/'checkpoints/best.pt'
    target_update=min(cfg['updates'],stop_after) if stop_after is not None else cfg['updates']
    if target_update<=step:raise ValueError('stop_after must exceed restored update')
    model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    regularizer=SIGReg(num_proj=cfg['sigreg_projections']).to(device)
    if resume:
        optimizer.load_state_dict(checkpoint['optimizer']);rng.bit_generator.state=checkpoint['sampler_state']
        torch.set_rng_state(checkpoint['torch_rng']);np.random.set_state(checkpoint['numpy_rng']);random.setstate(checkpoint['python_rng'])
        if device=='cuda' and checkpoint.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    before={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if v.is_floating_point() and not k.endswith(('running_mean','running_var'))}
    metadata={'config':cfg,'initialization':provenance,'action_stats':stats,'torch':torch.__version__,
              'device':device,'parameter_count':sum(p.numel() for p in model.parameters()),'signature':signature,
              'pretrained_weights_hash':sha256(Path(pretrained_dir)/'weights.pt') if pretrained_dir else None}
    if not resume:
        write_json(run/'config.json',metadata);write_json(run/'splits.json',read_json(splits_path))
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    def snapshot():
        return {'model':model.state_dict(),'model_config':model_cfg,'config':cfg,'action_stats':stats,
                'initialization':provenance,'optimizer':optimizer.state_dict(),'update':step,'best_validation_mse':best,
                'sampler_state':rng.bit_generator.state,'torch_rng':torch.get_rng_state(),
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else None,'signature':signature}
    with (run/'training.jsonl').open('a') as log:
        while step<target_update:
            begin=time.perf_counter();model.train()
            pixels,actions=make_batch(splits['model_train'],rng,cfg['batch_size'],stats,cfg['image_size'])
            pixels=pixels.to(device);actions=actions.to(device)
            # Explicit deterministic warmup/cosine schedule, recoverable from update.
            warmup=min(100,max(1,cfg['updates']//10))
            scale=min(1.,(step+1)/warmup)*.5*(1+math.cos(math.pi*max(0,step-warmup)/max(1,cfg['updates']-warmup)))
            for group in optimizer.param_groups:group['lr']=cfg['learning_rate']*scale
            optimizer.zero_grad(set_to_none=True)
            total,mse,reg,std=loss_values(model,pixels,actions,regularizer,cfg['sigreg_weight'])
            if not torch.isfinite(total):raise FloatingPointError('Nonfinite training loss')
            total.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('Nonfinite gradient')
            optimizer.step();step+=1
            row={'update':step,'loss':float(total.detach()),'prediction_mse':float(mse.detach()),'sigreg':float(reg.detach()),
                 'latent_std':float(std),'seconds':time.perf_counter()-begin,'lr':optimizer.param_groups[0]['lr']}
            if step%cfg['validate_every']==0 or step==target_update:
                model.eval();values=[]
                # Keep validation RNG consumption from changing subsequent training.
                cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None
                with torch.inference_mode():
                    vrng=np.random.default_rng(cfg['seed']+10000)
                    for _ in range(cfg['validation_batches']):
                        vp,va=make_batch(splits['model_val'],vrng,cfg['batch_size'],stats,cfg['image_size'])
                        vals=loss_values(model,vp.to(device),va.to(device),regularizer,cfg['sigreg_weight'])
                        values.append([float(x) for x in vals[1:]])
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
                row['validation_mse'],row['validation_sigreg'],row['validation_latent_std']=np.mean(values,0).tolist()
                if not np.isfinite(values).all():raise FloatingPointError('Nonfinite validation')
                is_best=row['validation_mse']<best and row['validation_latent_std']>1e-6
                if is_best:best=row['validation_mse']
                state=snapshot();save_checkpoint(last,state)
                if is_best:save_checkpoint(run/'checkpoints/best.pt',state)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    delta={k:float((v.detach().cpu()-before[k]).abs().max()) for k,v in model.state_dict().items() if k in before}
    summary={'updates':step,'configured_updates':cfg['updates'],'completed':step==cfg['updates'],'elapsed_seconds':time.perf_counter()-started,'changed_parameter_tensors':sum(v>0 for v in delta.values()),
             'max_parameter_change':max(delta.values()),'best_validation_mse':best,
             'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
             'purpose':cfg['purpose'],'claim':'pipeline execution only; no converged-memory claim for pilot'}
    write_json(run/'training_summary.json',summary)
    print(json.dumps(summary),flush=True)
    return run/'checkpoints/best.pt'
