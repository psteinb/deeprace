from deeprace.models.resnet import params_from_name, name
from deeprace.models import available

def test_is_available():

    res = available()

    assert res is not None
    assert isinstance(res,dict)

    assert len(res.keys()) > 0
    assert "resnet" in res.keys()

def test_available_has_content():

    res = available()

    assert "resnet" in res.keys()
    av_rnet = res["resnet"]
    assert len(av_rnet) == 3
    #print(av_rnet)
    assert len(av_rnet[0]) >= 1
    assert len(av_rnet[1]) >= 1
    assert len(av_rnet[-1]) >= 1

def test_defaults():

    tstr = name()
    res = params_from_name(tstr)

    assert res['n'] == 3
    assert res['version'] == 1


def test_resnet50():

    tstr = name(n=5, version=1)
    res = params_from_name(tstr)

    assert res['n'] == 5
    assert res['version'] == 1


def test_resnet50v2():

    tstr = name(n=5, version=2)
    res = params_from_name(tstr)

    assert res['n'] == 5
    assert res['version'] == 2


def test_resnet56v1_string():

    tstr = "resnet56v1"
    res = params_from_name(tstr)

    assert res['n'] == 9
    assert res['version'] == 1


def test_resnet56v2_string():

    tstr = "resnet56v2"
    res = params_from_name(tstr)

    assert res['n'] == 6
    assert res['version'] == 2


def test_resnet32v1_string():

    tstr = "resnet32v1"
    res = params_from_name(tstr)

    assert res['n'] == 5
    assert res['version'] == 1
