#!/usr/bin/env python3
import pytest

from deeprace.models import resnet


def can_load_keras():

    value = False
    try:
        import keras as K
    except Exception as ex:
        return value
    else:
        return True

    assert K.__version__


def test_constructs():

    rnet = resnet.model()

    assert rnet.num_classes == 10


def test_resnet_provides_something():

    rnet = resnet.model()
    obs = rnet.provides()

    assert len(obs) != 0
    assert "resnet32v1" in obs[0]
    assert "resnet56v1" in obs[0]


def test_resnet_has_options():

    rnet = resnet.model()
    obs = rnet.options()

    assert not isinstance(obs, type(None))
    assert isinstance(obs, type({}))

    assert obs["n"] == 5
    assert obs["num_classes"] == 10
    assert not "provides" in obs.keys()
    assert not "options" in obs.keys()


def test_resnet_has_dataloader():

    rnet = resnet.model()

    if can_load_keras():
        obs = rnet.data_loader(".")
        assert not isinstance(obs, type(None))
    else:
        with pytest.raises(Exception):
            rnet.data_loader(".")
