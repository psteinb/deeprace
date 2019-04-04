
__MODELS__ = {}

def available():
    """return a list of available models"""
    global __MODELS__
    return __MODELS__

def register_model(classifier, names=None, backends=None, datasets=None):
    global __MODELS__
    if __MODELS__ is None:
        __MODELS__ = { classifier: (names, backends, datasets)}
    else:
        if not classifier in __MODELS__.keys():
            __MODELS__[classifier] = (names, backends, datasets)

from deeprace.models.resnet import race
register_model("resnet",*race.provides())
