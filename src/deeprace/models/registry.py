#attempt to write a deeprace registry
#adopted from http://scottlobdell.me/2015/08/using-decorators-python-automatic-registration/

REGISTERED_RACES = {}

def registered_race(cls):
    """function that registers a class"""
    global REGISTERED_RACES
    REGISTERED_RACES[cls.__module__.split('.')[-1]] = cls.provides()
    return cls
