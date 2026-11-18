import numpy as np
import pytest
from memorymaze_lewm.segments import red_id, red_pixels, select_candidates


def test_target_color_and_position_identify_red_object_not_current_goal():
    d={'target_color':np.array([[.9,.1,.1],[.1,.9,.1]]),
       'target_pos':np.array([[2.,3.],[4.,5.]]),
       'targets_pos':np.array([[[2.,3.],[4.,5.],[6.,7.]]]*2)}
    assert red_id(d)==0
    d['target_pos'][0]=[4.,5.]
    with pytest.raises(ValueError):red_id(d)


def test_border_does_not_count_as_red_ball():
    x=np.zeros((3,64,64,3),np.uint8);x[:,:2, :,0]=255
    assert not red_pixels(x).any()
    x[:,30:34,30:34,0]=255
    assert np.all(red_pixels(x)==16)
