import torch


class AlternateModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.offset=torch.nn.Parameter(torch.tensor(2.))
    def encode(self,info):
        b,t=info['pixels'].shape[:2]
        return {'emb':torch.zeros(b,t,1,device=self.offset.device)}
    def action_encoder(self,a):return a
    def predict(self,z,a):return z+self.offset


def build(config):return AlternateModel()
