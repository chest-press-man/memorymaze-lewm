import numpy as np
import pytest
from memorymaze_lewm.data import make_window, preprocess, validate_splits


def episode(n=8):
    return {'image': np.zeros((n,64,64,3),np.uint8),
            'action': np.concatenate([np.zeros((1,6)),np.eye(6)[np.arange(n-1)%6]]),
            'reset': np.arange(n)==0, 'terminal': np.zeros(n,bool)}


def test_actions_are_outgoing_and_last_valid_window_kept():
    d=episode()
    images,actions=make_window(d,4)
    assert images.shape==(4,64,64,3)
    np.testing.assert_array_equal(actions,d['action'][5:8])
    with pytest.raises(ValueError): make_window(d,5)


def test_no_crossing_reset():
    d=episode();d['reset'][3]=True
    with pytest.raises(ValueError): make_window(d,1)


def test_border_color_removed_without_changing_interior():
    import torch
    x=np.full((1,64,64,3),100,np.uint8);y=x.copy()
    for a in (x,y):
        a[:,:2]=[255,0,0];a[:,-2:]=[255,0,0];a[:,:,:2]=[255,0,0];a[:,:,-2:]=[255,0,0]
    y[:,:2]=[0,255,0];y[:,-2:]=[0,255,0];y[:,:,:2]=[0,255,0];y[:,:,-2:]=[0,255,0]
    px=preprocess(x,224);py=preprocess(y,224)
    assert px.shape==(1,3,224,224)
    torch.testing.assert_close(px,py)


def test_split_leakage_is_rejected():
    with pytest.raises(ValueError): validate_splits({'train':['same.npz'],'test':['same.npz']})
