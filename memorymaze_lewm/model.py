"""Official LeWM construction and the model-replacement boundary."""
import copy
import importlib
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import ViTConfig, ViTModel

from .data import normalize_action, preprocess
from .io import read_json
from .vendor.lewm import LeWM
from .vendor.module import Embedder, MLP, Predictor


def create_model(cfg):
    if cfg.get('_model_factory'):
        clean_cfg=copy.deepcopy(cfg);target=clean_cfg.pop('_model_factory')
        module,name=target.split(':')
        model=getattr(importlib.import_module(module),name)(clean_cfg)
        if not isinstance(model,torch.nn.Module):raise TypeError('Model factory must return nn.Module')
        return model
    e=cfg['encoder']
    if e['size']!='tiny':raise ValueError('Only the documented ViT-tiny checkpoint schema is supported')
    vit=ViTModel(ViTConfig(hidden_size=192,num_hidden_layers=12,num_attention_heads=3,
                        intermediate_size=768,image_size=e['image_size'],patch_size=e['patch_size']),
                 add_pooling_layer=False,use_mask_token=e.get('use_mask_token',False))
    def clean(d):return {k:v for k,v in d.items() if not k.startswith('_') and k!='norm_fn'}
    return LeWM(vit,Predictor(**clean(cfg['predictor'])),Embedder(**clean(cfg['action_encoder'])),
                MLP(**clean(cfg['projector']),norm_fn=partial(torch.nn.BatchNorm1d)),
                MLP(**clean(cfg['pred_proj']),norm_fn=partial(torch.nn.BatchNorm1d)))


def initialize(pretrained_dir=None,model_factory=None):
    if pretrained_dir is None:
        cfg=read_json(Path(__file__).parent/'default_model.json')
        cfg['action_encoder']['input_dim']=6
        if model_factory:cfg['_model_factory']=model_factory
        return create_model(cfg),cfg,{'source':'scratch','reinitialized':['all']}
    cfg=read_json(Path(pretrained_dir)/'config.json')
    source=create_model(cfg)
    # Strict load before adaptation catches architecture/API drift.
    source.load_state_dict(torch.load(Path(pretrained_dir)/'weights.pt',map_location='cpu',weights_only=True),strict=True)
    old_dim=cfg['action_encoder']['input_dim'];cfg=copy.deepcopy(cfg);cfg['action_encoder']['input_dim']=6
    if old_dim!=6:source.action_encoder=Embedder(**{k:v for k,v in cfg['action_encoder'].items() if not k.startswith('_')})
    provenance={'source':str(Path(pretrained_dir).resolve()),'original_action_dim':old_dim,
                'reinitialized':['action_encoder'] if old_dim!=6 else []}
    if model_factory:
        cfg['_model_factory']=model_factory;target=create_model(cfg)
        old=source.state_dict();current=target.state_dict()
        shared={k:v for k,v in old.items() if k in current and current[k].shape==v.shape}
        if not shared:raise ValueError('No compatible pretrained keys; provide a compatible model factory')
        result=target.load_state_dict(shared,strict=False)
        provenance.update(transferred_keys=list(shared),new_keys=list(result.missing_keys),
                          unused_source_keys=[k for k in old if k not in shared])
        source=target
    return source,cfg,provenance


class LeWMAdapter:
    """rollout accepts INITIAL RGB only; actions and predictions advance history."""
    def __init__(self,model,config,stats):
        self.model=model.eval();self.config=config;self.stats=stats
        self.device=next(model.parameters()).device

    def reset_episode(self):
        reset=getattr(self.model,'reset_episode',None)
        if reset is not None:reset()

    @torch.inference_mode()
    def rollout(self,initial_images,past_actions,future_actions):
        self.reset_episode()
        history=self.config['history']
        if len(initial_images)!=history or past_actions.shape!=(history-1,6):raise ValueError('Invalid initial history')
        if future_actions.ndim!=2 or future_actions.shape[1]!=6 or len(future_actions)==0:raise ValueError('Invalid future actions')
        pixels=preprocess(initial_images,self.config['image_size']).unsqueeze(0).to(self.device)
        emb=self.model.encode({'pixels':pixels})['emb']
        acts=np.concatenate([past_actions,future_actions])
        acts=torch.from_numpy(normalize_action(acts,self.stats)).unsqueeze(0).to(self.device)
        all_actions=self.model.action_encoder(acts)
        states=list(emb.unbind(1));predictions=[]
        for t in range(len(future_actions)):
            lo=max(0,history+t-history)
            z=torch.stack(states[lo:],dim=1)
            a=all_actions[:,lo:history+t]
            nxt=self.model.predict(z,a)[:,-1]
            states.append(nxt);predictions.append(nxt)
        return torch.stack(predictions,dim=1)[0].cpu().numpy()


def load_adapter(checkpoint_path,device='cpu',factory=None):
    ckpt=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if factory:
        module,name=factory.split(':')
        return getattr(importlib.import_module(module),name)(ckpt,device)
    model=create_model(ckpt['model_config']);model.load_state_dict(ckpt['model'],strict=True);model.to(device)
    return LeWMAdapter(model,ckpt['config'],ckpt['action_stats'])
