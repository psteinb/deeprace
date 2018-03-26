from models.resnet import params_from_name, name

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
