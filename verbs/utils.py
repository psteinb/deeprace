import uuid
import yaml

def yaml_this(**kwargs):

    return yaml.dump(kwargs)


def uuid_from_this(*args):

    namespace = uuid.uuid1()

    value = uuid.uuid5(namespace," ".join([ str(item) for item in args ]))

    return value
