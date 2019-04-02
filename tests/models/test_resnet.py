from deeprace.models.resnet import params_from_name, name

def test_defaults():

    tstr = name()
    res = params_from_name(tstr)

    assert res['n'] == 3
    assert res['version'] == 1

def test_resnet50():

    tstr = name(n=5,version=1)
    res = params_from_name(tstr)

    assert res['n'] == 5
    assert res['version'] == 1

def test_resnet50v2():

    tstr = name(n=5,version=2)
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
