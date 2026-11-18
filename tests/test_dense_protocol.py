import json
from pathlib import Path

import numpy as np
import pytest


def test_scaled_split_keeps_holdouts_and_normalizer_fixed(tmp_path, monkeypatch):
    from memorymaze_lewm import data
    episodes=[{'path':f'/old/train/{i:04}.npz','source':'train'} for i in range(2500)]
    episodes += [{'path':f'/old/eval/e{i:04}.npz','source':'eval'} for i in range(200)]
    (tmp_path/'acquisition.json').write_text(json.dumps({'episodes':episodes}))
    monkeypatch.setattr(data,'cached_episode',lambda p:{'maze_layout':np.ones((1,2,2),np.uint8)})
    small=data.make_splits(tmp_path,counts=(500,100,300,100,200),protocol=True)
    large=data.make_splits(tmp_path,counts=(2000,100,300,100,200),protocol=True)
    assert small['counts']==dict(model_train=500,model_val=100,probe_train=300,probe_val=100,test=200)
    for key in ['model_val','probe_train','probe_val','test']:
        assert small['splits'][key]==large['splits'][key]
    assert large['splits']['model_train'][:500]==small['splits']['model_train']
    assert small['normalizer_train']==large['normalizer_train']==small['splits']['model_train']
    data.validate_splits(large['splits'])


@pytest.mark.parametrize('max_step',[50,70])
def test_extract_keeps_observed_zero_and_every_future_step(tmp_path,monkeypatch,max_step):
    from memorymaze_lewm import rollout
    class Adapter:
        def initial_latent(self,images):
            assert images.shape==(3,64,64,3)
            return np.array([0.,10.])
        def rollout(self,images,past_actions,future_actions):
            assert len(images)==3 and len(past_actions)==2
            assert len(future_actions)==max_step
            return np.stack([np.arange(1,max_step+1),np.arange(1,max_step+1)+10],axis=-1)
    d={'image':np.zeros((100,64,64,3),np.uint8),'action':np.eye(6)[np.arange(100)%6],
       'targets_vec':np.broadcast_to(np.arange(100)[:,None,None],(100,3,2))}
    segs=[{'review_status':'approved','path':'fake.npz','origin':5,'object_id':0,'episode_id':s,
           'segment_id':s+':5','split':s,'visibility':['visible']*(max_step+1)}
          for s in ['probe_train','probe_val','test']]
    path=tmp_path/'segments.json';path.write_text(json.dumps({'segments':segs}))
    ckpt=tmp_path/'model.pt';ckpt.write_bytes(b'fixture')
    monkeypatch.setattr(rollout,'cached_episode',lambda p:d)
    monkeypatch.setattr(rollout,'load_adapter',lambda *args:Adapter())
    rollout.extract(ckpt,path,tmp_path/'out',max_step=max_step)
    with np.load(tmp_path/'out/test.npz') as f:
        np.testing.assert_array_equal(f['horizon'],np.arange(max_step+1))
        np.testing.assert_array_equal(f['source_step'],np.arange(5,6+max_step))
        np.testing.assert_array_equal(f['latent'][:,0],np.arange(max_step+1))
        assert f['latent_source'][0]=='observed' and np.all(f['latent_source'][1:]=='predicted')


def test_probe_uses_dense_zero_reference_and_validation_only(tmp_path):
    from memorymaze_lewm.evaluate import run_probe
    root=tmp_path/'latent';root.mkdir()
    rng=np.random.default_rng(42)
    for split in ['probe_train','probe_val','test']:
        hs=np.tile(np.arange(51),3);ep=np.repeat([split+str(i) for i in range(3)],51)
        z=rng.normal(size=(len(hs),4));target=z[:,:2]*2+1
        np.savez_compressed(root/f'{split}.npz',latent=z,target=target,episode=ep,segment=ep,
                            horizon=hs,source_step=hs+3,object_id=np.zeros(len(hs),int),
                            visibility=np.full(len(hs),'hidden'),review_status=np.full(len(hs),'approved'))
    (root/'provenance.json').write_text('{}')
    metrics=run_probe(root,tmp_path/'probe')
    assert list(metrics['model'])==[str(i) for i in range(51)]
    assert metrics['increase_from_step0']['0']['mean_l2']==0
    assert metrics['model']['50']['mean_l2']<1e-7
    assert metrics['horizon_zero_kind']=='observed initial latent; not a future prediction'
    assert 'validation_decline' in metrics


def test_checkpoint_variance_gate_rejects_collapsed_or_nonfinite_states():
    from memorymaze_lewm.train import checkpoint_eligible
    assert checkpoint_eligible(.2,.3,1.)
    assert not checkpoint_eligible(.001,.09,1.)
    assert not checkpoint_eligible(float('nan'),.3,1.)
    assert not checkpoint_eligible(.2,float('nan'),1.)


def test_fixed_validation_index_covers_every_episode_reproducibly(monkeypatch):
    from memorymaze_lewm import train
    d={'image':np.zeros((8,64,64,3),np.uint8),'action':np.eye(6)[np.arange(8)%6],
       'reset':np.arange(8)==0,'terminal':np.zeros(8,bool)}
    monkeypatch.setattr(train,'cached_episode',lambda p:d)
    paths=[f'val{i}.npz' for i in range(100)]
    a=train.fixed_validation_index(paths,10042);b=train.fixed_validation_index(paths,10042)
    assert a==b and [e['path'] for e in a]==paths
    assert all(0<=e['start']<=4 for e in a)


def test_real_adapter_zero_is_last_initial_observation():
    from tests.test_model import RecordingModel
    from memorymaze_lewm.model import LeWMAdapter
    adapter=LeWMAdapter(RecordingModel(),{'history':3,'image_size':224},{'mean':[0]*6,'std':[1]*6})
    zero=adapter.initial_latent(np.zeros((3,64,64,3),np.uint8))
    np.testing.assert_array_equal(zero,[20.])
