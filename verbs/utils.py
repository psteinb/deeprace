import uuid
import yaml
#from containers import OrderedDict

def yaml_this(**kwargs):
    """ store the input keyword arguments <kwargs> in a yaml compatible format, here the order is NOT preserved in the yaml output"""
    return yaml.dump(kwargs, default_flow_style=False)

def yaml_ordered(odict):
    """ expect a collections.OrderedDict object as input, if so the order is preserved in the yaml output"""
    value = ""

    for k,v in odict.items():
        value += yaml.dump({k : v}, default_flow_style=False)
    return value



def uuid_from_this(*args):

    namespace = uuid.uuid1()

    value = uuid.uuid5(namespace," ".join([ str(item) for item in args ]))

    return value
