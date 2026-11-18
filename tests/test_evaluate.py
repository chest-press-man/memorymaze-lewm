import numpy as np
from memorymaze_lewm.evaluate import episode_metrics


def test_multiple_segments_do_not_give_one_episode_more_weight():
    result=episode_metrics(np.array([0.,0.,9.]),np.array(['a','a','b']),np.ones(3,int))
    assert result['1']['mean_l2']==4.5
    assert result['1']['episodes']==2
    assert result['1']['segments']==3
    assert result['3']['mean_l2'] is None
