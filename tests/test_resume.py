from pathlib import Path
import pytest


def test_actual_training_resumes_optimizer_and_update(tmp_path):
    import torch
    from memorymaze_lewm.io import read_json,write_json
    from memorymaze_lewm.train import train
    root=Path(__file__).resolve().parents[1]
    if not (root/'data/splits.json').exists() or not (root/'data/pretrained/cube/weights.pt').exists():
        pytest.skip('Acquire data and fetch official Cube weights for the training integration test')
    cfg=read_json(root/'configs/pilot.json');cfg.update(updates=4,validate_every=2,validation_batches=1)
    config=tmp_path/'config.json';write_json(config,cfg);run=tmp_path/'resume'
    train(config,root/'data/splits.json',run,root/'data/pretrained/cube',stop_after=2)
    first=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert first['update']==2
    train(config,root/'data/splits.json',run,resume=True)
    last=torch.load(run/'checkpoints/last.pt',map_location='cpu',weights_only=False)
    assert last['update']==4
    assert int(next(iter(last['optimizer']['state'].values()))['step'])==4
    original=(run/'training_summary.json').read_bytes()
    train(config,root/'data/splits.json',run,resume=True)
    assert (run/'training_summary.json').read_bytes()==original
