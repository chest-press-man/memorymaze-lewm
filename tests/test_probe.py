import numpy as np
from memorymaze_lewm.probe import fit_ridge, predict, balanced_weights


def test_linear_readout_generalizes_and_handles_constant_feature():
    rng=np.random.default_rng(42)
    x=rng.normal(size=(100,4));x[:,3]=1
    y=x[:,:2]@np.array([[2.,1.],[-1.,3.]])+[4.,5.]
    m=fit_ridge(x[:60],y[:60],np.ones(60)/60,0)
    np.testing.assert_allclose(predict(m,x[60:]),y[60:],atol=1e-8)


def test_each_episode_and_horizon_get_equal_weight():
    ids=np.array(['a','a','b','b','b','b'])
    h=np.array([1,3,1,3,1,3])
    w=balanced_weights(ids,h)
    assert np.isclose(w[ids=='a'].sum(),.5)
    assert np.isclose(w[ids=='b'].sum(),.5)
    assert np.isclose(w[h==1].sum(),.5)
