import numpy as np
import torch
from memorymaze_lewm.model import LeWMAdapter, create_model


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.tensor(0.));self.calls=[]
    def encode(self,info):
        return {'emb':torch.tensor([[[18.],[19.],[20.]]],device=self.anchor.device)}
    def action_encoder(self,a): return a
    def predict(self,z,a):
        self.calls.append((z.detach().clone(),a.detach().clone()))
        return z+1


def test_rollout_only_uses_initial_pixels_then_predicted_history():
    m=RecordingModel();a=LeWMAdapter(m,{'history':3,'image_size':224}, {'mean':[0]*6,'std':[1]*6})
    out=a.rollout(np.zeros((3,64,64,3),np.uint8),np.eye(6)[:2],np.eye(6)[:4])
    np.testing.assert_allclose(out[:,0],[21,22,23,24])
    torch.testing.assert_close(m.calls[1][0],torch.tensor([[[19.],[20.],[21.]]]))
    assert len(m.calls)==4


def test_fresh_adapter_resets_for_each_segment():
    m=RecordingModel();a=LeWMAdapter(m,{'history':3,'image_size':224},{'mean':[0]*6,'std':[1]*6})
    args=(np.zeros((3,64,64,3),np.uint8),np.eye(6)[:2],np.eye(6)[:1])
    np.testing.assert_array_equal(a.rollout(*args),a.rollout(*args))


def test_official_predictor_does_not_read_future_positions_in_eval():
    from memorymaze_lewm.vendor.module import Predictor
    m=Predictor(num_frames=3,depth=1,heads=2,mlp_dim=32,input_dim=12,hidden_dim=12,dim_head=6).eval()
    z=torch.randn(2,3,12);a=torch.randn(2,3,12)
    changed=z.clone();changed[:,1:]+=100
    with torch.no_grad():
        torch.testing.assert_close(m(z,a)[:,0],m(changed,a)[:,0])


def test_alternate_model_checkpoint_uses_same_rollout_boundary(tmp_path):
    from memorymaze_lewm.model import load_adapter
    cfg={'_model_factory':'tests.model_factories:build'}
    m=create_model(cfg);path=tmp_path/'alternate.pt'
    torch.save({'model':m.state_dict(),'model_config':cfg,'config':{'history':3,'image_size':224},
                'action_stats':{'mean':[0]*6,'std':[1]*6}},path)
    adapter=load_adapter(path)
    out=adapter.rollout(np.zeros((3,64,64,3),np.uint8),np.eye(6)[:2],np.eye(6)[:4])
    np.testing.assert_allclose(out[:,0],[2,4,6,8])
