from deeprace.models.registry import REGISTERED_RACES

# def register_model(classifier, names=None, backends=None, datasets=None):
#     global __MODELS__
#     if __MODELS__ is None:
#         __MODELS__ = { classifier: (names, backends, datasets)}
#     else:
#         if not classifier in __MODELS__.keys():
#             __MODELS__[classifier] = (names, backends, datasets)

# from deeprace.models.resnet import race
# register_model("resnet",*race.provides())

#Import all classes in this directory so that classes with @register_class are registered.

__import__("deeprace.models.resnet", globals(), locals())


# __all__ = [
#     'REGISTERED_RACES'
# ]

def available():
    """return a list of available models"""

    return REGISTERED_RACES
